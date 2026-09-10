"""LSTM baseline per Jakut station (split B_clean, horizons 1/6/12/24) reusing
experiments/lstm_baseline.py. Answers RQ3 lead time per station.

    python3 experiments/lstm_multistation.py [--quick]

Output: reports/lstm_multistation/metrics.json + metrics.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_baseline as lb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/lstm_multistation"
# Most Jakut stations only start Nov 2023, so the node-140 split (train 2021-23)
# is unusable here. Common split for all stations; E1 (Jan 2025) lands in val,
# E2 (Jan 2026) in test. keep_flags=True: at 126/166/169/170 the hourly qc_flag=8
# rate is 44-54 %, so requiring a clean 72 h lookback (B_clean) leaves ~0 windows.
# Metrics on flag-0 targets are still reported (*_flag0 columns).
# ponytail: point-QC thresholds were tuned on station 140; re-tune per station
# type (bubble vs pressure) before trusting flag-8 as "bad" elsewhere.
SPLIT = (("2023-11-01", "2024-12-31"), ("2025-01-01", "2025-04-30"), ("2025-05-01", "2026-12-31"), True)


def run_station(df: pd.DataFrame, horizons, epochs, device) -> list[dict]:
    rows = []
    train_r, val_r, test_r, keep_flags = SPLIT
    df = lb.prepare(df)
    for h in horizons:
        win = lb.make_windows(df, horizon=h)
        wt = win["waktu"]
        base = np.ones(len(win["y"]), bool) if keep_flags else (win["qc_any"] == 0)
        tr_m, va_m, te_m = (base & np.asarray((wt >= pd.Timestamp(r[0])) & (wt <= pd.Timestamp(r[1])))
                            for r in (train_r, val_r, test_r))
        if tr_m.sum() < 2000 or va_m.sum() < 100 or te_m.sum() < 500:
            rows.append({"horizon": h, "n_train": int(tr_m.sum()), "n_val": int(va_m.sum()), "n_test": int(te_m.sum()),
                         "note": "too few windows (126: no data Jan-Sep 2025 = val period)"})
            continue
        r = lb.fit_eval(win, tr_m, va_m, te_m, epochs, device)
        yhat = r.pop("yhat")
        pd.DataFrame(dict(waktu=wt[te_m], y=win["y"][te_m], yhat=yhat, qc_flag=win["qc_flag"][te_m])).to_parquet(
            OUT / f"pred_{df.attrs.get('sid', 'x')}_h{h}.parquet", index=False)
        rows.append({"horizon": h, **r})
    return rows


def main(quick: bool) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    horizons, epochs = ([1, 24], 2) if quick else ([1, 6, 12, 24], 30)
    multi = pd.read_parquet(ROOT / "forcing-acquisition/data/processed/forcing_hourly_multi.parquet")
    multi["waktu"] = pd.to_datetime(multi["waktu"], utc=True).dt.tz_convert(None)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for sid, g in multi.groupby("stasiun_id"):
        name = g["stasiun_nama"].iloc[0]
        gdf = g.drop(columns=["stasiun_id", "stasiun_nama"]).reset_index(drop=True)
        gdf.attrs["sid"] = sid
        for r in run_station(gdf, horizons, epochs, device):
            results.append({"stasiun_id": sid, "stasiun_nama": name, **r})
        print(f"{sid} {name}: done ({time.time() - t0:.0f}s)")
    (OUT / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    m = pd.DataFrame(results)
    m.to_csv(OUT / "metrics.csv", index=False)
    print(m.pivot(index=["stasiun_id", "stasiun_nama"], columns="horizon", values="lstm_nse").round(3).to_string())


if __name__ == "__main__":
    main("--quick" in sys.argv)
