"""Compare LSTM configs: rain source (CHIRPS daily vs GSMaP hourly) x loss
(mse vs quantile) on the multistation reruns from lstm_multistation.py.

    python3 experiments/compare_rain_loss.py

Reads whichever of the 3 report dirs exist and says which are missing (runs
fine with only chirps_mse present, e.g. before the GSMaP reruns are done).
Output: reports/compare_rain_loss/metrics.csv + summary.md
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/compare_rain_loss"
CONFIGS = {
    "chirps_mse": ROOT / "reports/lstm_multistation",
    "gsmap_mse": ROOT / "reports/lstm_multistation_gsmap_mse",
    "gsmap_quantile": ROOT / "reports/lstm_multistation_gsmap_quantile",
}
COLS = ["stasiun_id", "stasiun_nama", "horizon", "lstm_nse", "lstm_rmse", "f1_p95", "pod_p95", "far_p95"]


def load(config: str, path: Path) -> pd.DataFrame | None:
    mp = path / "metrics.csv"
    if not mp.exists():
        return None
    df = pd.read_csv(mp)
    have = [c for c in COLS if c in df.columns]
    df = df[have].copy()
    df["config"] = config
    return df


def main() -> None:
    frames, missing = [], []
    for config, path in CONFIGS.items():
        df = load(config, path)
        if df is None:
            missing.append(config)
            print(f"missing: {config} ({path}/metrics.csv not found)")
        else:
            frames.append(df)
    if not frames:
        print("no report dirs found - nothing to compare")
        return
    combined = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "metrics.csv", index=False)

    piv = combined.pivot_table(index=["stasiun_id", "stasiun_nama", "horizon"], columns="config", values="lstm_nse")
    lines = [
        "# Rain source x loss comparison\n",
        f"Configs present: {', '.join(sorted(combined['config'].unique()))}",
        f"Configs missing: {', '.join(missing) if missing else 'none'}\n",
        "NSE by config, station x horizon:\n", "```", piv.round(3).to_string(), "```",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"present: {sorted(combined['config'].unique())}, missing: {missing}")
    print(f"saved {OUT}/metrics.csv, summary.md")


if __name__ == "__main__":
    main()
