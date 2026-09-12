"""LSTM baseline per Jakut station (split B_clean, horizons 1/6/12/24) reusing
experiments/lstm_baseline.py. Answers RQ3 lead time per station.

    python3 experiments/lstm_multistation.py [--quick] [--station ID] [--rain chirps|gsmap] [--loss mse|weighted|quantile]

--station ID: run only that station and merge its rows into the existing
metrics.json/csv (other stations' rows are kept as-is).
--rain/--loss: default chirps/mse -> output dir unchanged (reports/lstm_multistation/);
any other combo -> reports/lstm_multistation_{rain}_{loss}/, so the default
run's output path/behavior is untouched.

Also scores F1/POD/FAR at the per-station train-p95 threshold (same
definition as experiments/anomaly_f1.py) alongside NSE, so the comparison
table has NSE + event-detection metrics without a separate script.

Output: metrics.json + metrics.csv under the output dir above.
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
from anomaly_f1 import confusion_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out_dir(rain: str, loss: str) -> Path:
    if rain == "chirps" and loss == "mse":
        return ROOT / "reports/lstm_multistation"  # unchanged default path
    return ROOT / f"reports/lstm_multistation_{rain}_{loss}"
# Most Jakut stations only start Nov 2023, so the node-140 split (train 2021-23)
# is unusable here. Common split for all stations; E1 (Jan 2025) lands in val,
# E2 (Jan 2026) in test. keep_flags=True: at 126/166/169/170 the hourly qc_flag=8
# rate is 44-54 %, so requiring a clean 72 h lookback (B_clean) leaves ~0 windows.
# Metrics on flag-0 targets are still reported (*_flag0 columns).
# ponytail: point-QC thresholds were tuned on station 140; re-tune per station
# type (bubble vs pressure) before trusting flag-8 as "bad" elsewhere.
SPLIT = (("2023-11-01", "2024-12-31"), ("2025-01-01", "2025-04-30"), ("2025-05-01", "2026-12-31"), True)
# 126's raw scrape has a gap 2024-11-15..2025-11-01 (source outage), which
# swallows the shared val window whole (0 val windows -> skipped, NaN in
# metrics.csv). Val/test shifted forward to land after the gap; train
# unchanged (unaffected - ends before the gap starts).
SPLIT_OVERRIDES = {
    126: (("2023-11-01", "2024-12-31"), ("2025-11-01", "2025-12-31"), ("2026-01-01", "2026-12-31"), True),
}


def run_station(df: pd.DataFrame, horizons, epochs, device, sid: int, out: Path, loss: str) -> list[dict]:
    rows = []
    train_r, val_r, test_r, keep_flags = SPLIT_OVERRIDES.get(sid, SPLIT)
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
        r = lb.fit_eval(win, tr_m, va_m, te_m, epochs, device, loss={"weighted": "weighted_mse"}.get(loss, loss))
        yhat = r.pop("yhat")
        # F1/POD/FAR at train-p95 threshold (anomaly_f1.py convention), flag-0 targets only.
        p95_thr = float(np.percentile(win["y"][tr_m], 95))
        f0 = win["qc_flag"][te_m] == 0
        cm = confusion_metrics(win["y"][te_m][f0] >= p95_thr, yhat[f0] >= p95_thr)
        r.update({"f1_p95": cm["f1"], "pod_p95": cm["pod"], "far_p95": cm["far"]})
        pd.DataFrame(dict(waktu=wt[te_m], y=win["y"][te_m], yhat=yhat, qc_flag=win["qc_flag"][te_m])).to_parquet(
            out / f"pred_{df.attrs.get('sid', 'x')}_h{h}.parquet", index=False)
        rows.append({"horizon": h, **r})
    return rows


def main(quick: bool, only_station: int | None, rain: str, loss: str) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    horizons, epochs = ([1, 24], 2) if quick else ([1, 6, 12, 24], 30)
    parquet_name = "forcing_hourly_multi.parquet" if rain == "chirps" else "forcing_hourly_multi_gsmap.parquet"
    ppath = ROOT / "forcing-acquisition/data/processed" / parquet_name
    if not ppath.exists():
        raise SystemExit(f"--rain {rain} needs {ppath} (run harmonize_multi.py --rain {rain} first)")
    multi = pd.read_parquet(ppath)
    multi["waktu"] = pd.to_datetime(multi["waktu"], utc=True).dt.tz_convert(None)
    multi["stasiun_id"] = multi["stasiun_id"].astype(int)
    if rain == "gsmap" and multi["waktu"].min() > pd.Timestamp("2023-11-01"):
        raise SystemExit(f"--rain gsmap parquet only starts {multi['waktu'].min()}, split needs 2023-11-01")
    if only_station is not None:
        multi = multi[multi["stasiun_id"] == only_station]
    out = out_dir(rain, loss)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for sid, g in multi.groupby("stasiun_id"):
        name = g["stasiun_nama"].iloc[0]
        gdf = g.drop(columns=["stasiun_id", "stasiun_nama"]).reset_index(drop=True)
        gdf.attrs["sid"] = sid
        for r in run_station(gdf, horizons, epochs, device, sid, out, loss):
            results.append({"stasiun_id": sid, "stasiun_nama": name, **r})
        print(f"{sid} {name}: done ({time.time() - t0:.0f}s)")

    if only_station is not None and (out / "metrics.json").exists():
        prior = json.loads((out / "metrics.json").read_text())
        results = [r for r in prior if int(r["stasiun_id"]) != only_station] + results
    for r in results:
        r["stasiun_id"] = int(r["stasiun_id"])

    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    m = pd.DataFrame(results)
    m.to_csv(out / "metrics.csv", index=False)
    print(m.pivot(index=["stasiun_id", "stasiun_nama"], columns="horizon", values="lstm_nse").round(3).to_string())


if __name__ == "__main__":
    station_arg = None
    if "--station" in sys.argv:
        station_arg = int(sys.argv[sys.argv.index("--station") + 1])
    rain_arg = sys.argv[sys.argv.index("--rain") + 1] if "--rain" in sys.argv else "chirps"
    loss_arg = sys.argv[sys.argv.index("--loss") + 1] if "--loss" in sys.argv else "mse"
    main("--quick" in sys.argv, station_arg, rain_arg, loss_arg)
