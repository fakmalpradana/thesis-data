"""LSTM baseline for TMA forecasting at station 140 (node 140 forcing table),
plus a train/val/test split decision: 3 split configs x 4 horizons, LSTM vs
persistence, NSE/RMSE/MAE on all test rows and on flag-0-only test rows.

    python3 experiments/lstm_baseline.py [--quick]

Writes reports/lstm_baseline/metrics.json and pred_{config}_h{h}.parquet.
"""

import argparse
import json
import random
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "lstm_baseline"
LOOKBACK = 72

# (train_range, val_range, test_range, keep_flags) — keep_flags=False drops any
# window (history or target) touching tma_qc_flag != 0.
SPLITS = {
    "A_chrono": (("2021-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31"), True),
    "B_clean": (("2021-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31"), False),
    "C_pre2024": (("2021-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), True),
}


def load() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
    df = con.sql("SELECT * FROM forcing_hourly ORDER BY waktu").df()
    con.close()
    df["waktu"] = pd.to_datetime(df["waktu"], utc=True).dt.tz_localize(None)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tma_mean"] = df["tma_mean"].interpolate(method="linear", limit=6)
    return df


def make_windows(df: pd.DataFrame, lookback: int = LOOKBACK, horizon: int = 1) -> dict:
    """Sliding windows: history = [t-lookback+1 .. t], target = t+horizon.

    X: (n, lookback, 3) of [tma_mean, tide_m, hujan_mm] over history (rain at
    t+h is NOT included — hujan_mm is daily-constant, so a future value would
    leak same-day rain). tide_future = tide_m[t+horizon] (astronomical, known
    in advance -> not leakage). y = tma_mean[t+horizon]. y_last = tma_mean[t]
    (persistence prediction). Windows with any NaN are dropped.
    """
    tma = df["tma_mean"].to_numpy(float)
    tide = df["tide_m"].to_numpy(float)
    hujan = df["hujan_mm"].to_numpy(float)
    flag = df["tma_qc_flag"].to_numpy(float)
    waktu = df["waktu"].to_numpy()
    n = len(df)
    n_win = n - lookback - horizon + 1
    if n_win <= 0:
        raise ValueError("series too short for lookback+horizon")

    start = np.arange(n_win)
    hist_idx = start[:, None] + np.arange(lookback)[None, :]
    t_idx = start + lookback - 1
    tgt_idx = t_idx + horizon

    X = np.stack([tma[hist_idx], tide[hist_idx], hujan[hist_idx]], axis=-1)
    tide_future = tide[tgt_idx]
    y = tma[tgt_idx]
    y_last = tma[t_idx]
    qc_target = flag[tgt_idx]
    qc_any = np.fmax(np.nanmax(flag[hist_idx], axis=1), qc_target)

    valid = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(tide_future) | np.isnan(y) | np.isnan(y_last))
    return dict(
        X=X[valid], tide_future=tide_future[valid], y=y[valid], y_last=y_last[valid],
        waktu=pd.to_datetime(waktu[tgt_idx][valid]), qc_flag=qc_target[valid], qc_any=qc_any[valid],
    )


def nse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.sum((y - y.mean()) ** 2)
    return float("nan") if denom == 0 else 1 - np.sum((y - yhat) ** 2) / denom


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return dict(
        nse=nse(y, yhat),
        rmse=float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else float("nan"),
        mae=float(np.mean(np.abs(y - yhat))) if len(y) else float("nan"),
        n=int(len(y)),
    )


def persistence_baseline(y_last, y) -> dict:
    return metrics(y, y_last)


class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden + 1, 1)

    def forward(self, seq, tide_future):
        _, (h, _) = self.lstm(seq)
        z = torch.cat([h[-1], tide_future.unsqueeze(-1)], dim=-1)
        return self.fc(z).squeeze(-1)


def train(X_tr, tide_tr, y_tr, X_va, tide_va, y_va, epochs=30, patience=5, lr=1e-3, seed=42, device="cpu"):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = LSTMModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ttr_t = torch.tensor(tide_tr, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    Tva_t = torch.tensor(tide_va, dtype=torch.float32, device=device)
    Yva_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    ds = torch.utils.data.TensorDataset(Xtr_t, Ttr_t, Ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, tb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb, tb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t, Tva_t), Yva_t).item()
        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def fit_eval(win, tr_m, va_m, te_m, epochs, device):
    """Train on tr_m, early-stop on va_m, score on te_m. Returns metrics dict + yhat."""
    fmean = win["X"][tr_m].reshape(-1, 3).mean(axis=0)
    fstd = win["X"][tr_m].reshape(-1, 3).std(axis=0) + 1e-8
    ymean, ystd = win["y"][tr_m].mean(), win["y"][tr_m].std() + 1e-8

    def norm_X(m):
        return (win["X"][m] - fmean) / fstd

    def norm_T(m):
        return (win["tide_future"][m] - fmean[1]) / fstd[1]

    try:
        model = train(
            norm_X(tr_m), norm_T(tr_m), (win["y"][tr_m] - ymean) / ystd,
            norm_X(va_m), norm_T(va_m), (win["y"][va_m] - ymean) / ystd,
            epochs=epochs, device=device,
        )
    except (RuntimeError, NotImplementedError):
        device = "cpu"
        model = train(
            norm_X(tr_m), norm_T(tr_m), (win["y"][tr_m] - ymean) / ystd,
            norm_X(va_m), norm_T(va_m), (win["y"][va_m] - ymean) / ystd,
            epochs=epochs, device=device,
        )

    model.eval()
    with torch.no_grad():
        yhat_n = model(
            torch.tensor(norm_X(te_m), dtype=torch.float32, device=device),
            torch.tensor(norm_T(te_m), dtype=torch.float32, device=device),
        ).cpu().numpy()
    yhat = yhat_n * ystd + ymean
    y_te, ylast_te, flag_te = win["y"][te_m], win["y_last"][te_m], win["qc_flag"][te_m]
    f0 = flag_te == 0

    lstm_all, lstm_f0 = metrics(y_te, yhat), metrics(y_te[f0], yhat[f0])
    pers_all, pers_f0 = persistence_baseline(ylast_te, y_te), persistence_baseline(ylast_te[f0], y_te[f0])

    res = dict(
        n_train=int(tr_m.sum()), n_val=int(va_m.sum()), n_test=int(te_m.sum()), n_test_flag0=int(f0.sum()),
        lstm_nse=lstm_all["nse"], lstm_rmse=lstm_all["rmse"], lstm_mae=lstm_all["mae"],
        lstm_nse_flag0=lstm_f0["nse"], lstm_rmse_flag0=lstm_f0["rmse"], lstm_mae_flag0=lstm_f0["mae"],
        persistence_nse=pers_all["nse"], persistence_rmse=pers_all["rmse"], persistence_mae=pers_all["mae"],
        persistence_nse_flag0=pers_f0["nse"], persistence_rmse_flag0=pers_f0["rmse"], persistence_mae_flag0=pers_f0["mae"],
        )
    res["yhat"] = yhat
    return res


def run(quick: bool):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # ponytail: MPS LSTM support is uneven across torch versions; fall back to
    # CPU (dataset is small, CPU is plenty fast) if a training step fails.
    df = prepare(load())
    horizons = [1, 24] if quick else [1, 6, 12, 24]
    epochs = 2 if quick else 30

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in horizons:
        win = make_windows(df, horizon=h)
        wt = win["waktu"]
        for name, (train_range, val_range, test_range, keep_flags) in SPLITS.items():
            mask = np.ones(len(win["y"]), bool) if keep_flags else (win["qc_any"] == 0)

            def in_range(rng):
                return np.asarray((wt >= pd.Timestamp(rng[0])) & (wt <= pd.Timestamp(rng[1])))

            tr_m = mask & in_range(train_range)
            va_m = mask & in_range(val_range)
            te_m = mask & in_range(test_range)
            if tr_m.sum() < 50 or va_m.sum() < 10 or te_m.sum() < 10:
                continue

            res = fit_eval(win, tr_m, va_m, te_m, epochs, device)
            yhat = res.pop("yhat")
            y_te, flag_te = win["y"][te_m], win["qc_flag"][te_m]
            rows.append(dict(config=name, horizon=h, **res))

            pred_df = pd.DataFrame(dict(waktu=wt[te_m], y=y_te, yhat=yhat, qc_flag=flag_te))
            pred_df.to_parquet(REPORT_DIR / f"pred_{name}_h{h}.parquet", index=False)

    (REPORT_DIR / "metrics.json").write_text(json.dumps(rows, indent=2))
    return rows


def print_table(rows):
    df = pd.DataFrame(rows)
    cols = ["config", "horizon", "n_test", "lstm_nse", "persistence_nse", "n_test_flag0", "lstm_nse_flag0", "persistence_nse_flag0"]
    print(df[cols].to_string(index=False))


def _self_check():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(nse(y, y) - 1.0) < 1e-9
    assert abs(nse(y, np.full_like(y, y.mean()))) < 1e-9

    tiny = pd.DataFrame(dict(
        waktu=pd.date_range("2021-01-01", periods=10, freq="h"),
        tma_mean=np.arange(10.0), tide_m=np.arange(10.0) * 0.1, hujan_mm=np.zeros(10), tma_qc_flag=np.zeros(10),
    ))
    w = make_windows(tiny, lookback=3, horizon=1)
    assert w["X"].shape == (7, 3, 3), w["X"].shape
    assert len(w["y"]) == 7 and len(w["waktu"]) == 7


if __name__ == "__main__":
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    rows = run(args.quick)
    print(f"done in {time.time() - t0:.1f}s")
    print_table(rows)
