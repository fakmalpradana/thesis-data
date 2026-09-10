"""Multi-station forcing table for all Jakut TMA stations: tma_hourly (per
station) joined to the node-140 rain + tide series.

    python3 forcing-acquisition/src/harmonize_multi.py

# ponytail: one rain cell + one tide point shared by all stations. CHIRPS is
# 5 km and the bay is ~20 km wide, EOT20 is 14 km - per-station forcing would
# be the same cell anyway. Upgrade path: per-station CHIRPS/GSMaP cell lookup
# via stations.lat/lon once sub-daily rain exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # data/
JAKUT = [107, 140, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187, 150, 126]


def main() -> None:
    f = pd.read_parquet(ROOT / "forcing-acquisition/data/processed/forcing_hourly.parquet")
    forcing = f[["waktu", "hujan_mm", "hujan_qc_flag", "tide_m", "tide_qc_flag"]]
    files = [p for p in (ROOT / "tinggi_air/data/processed").glob("*_hourly.parquet")
             if int(p.name.split("_")[0]) in JAKUT]
    tma = pd.concat(pd.read_parquet(p) for p in files)
    tma = tma.rename(columns={"n_obs": "tma_n_obs", "pct_flagged": "tma_pct_flagged", "qc_flag": "tma_qc_flag"})
    out = tma.merge(forcing, on="waktu", how="inner").sort_values(["stasiun_id", "waktu"])
    # Hourly spike filter: |tma - rolling 24h median| > 150 cm is physically
    # impossible for these gates/sumps (seen: -6047 cm at 166, -820 at 179).
    # Point-level QC in tinggi_air/src/qc.py misses them because a whole
    # hour of raw points is wrong, not a lone outlier. Flag 7, value -> NaN.
    # ponytail: fixed 150 cm threshold; make it per-station (e.g. 3x siaga range) if it bites real floods.
    med = out.groupby("stasiun_id")["tma_mean"].transform(lambda x: x.rolling(25, center=True, min_periods=5).median())
    spike = ((out["tma_mean"] - med).abs() > 150) | ~out["tma_mean"].between(-300, 1000)  # plateaus of -6047 at 166
    out.loc[spike, "tma_qc_flag"] = 7
    out.loc[spike, "tma_mean"] = np.nan
    print(f"spike-flagged hours (flag 7): {int(spike.sum())}")
    p = ROOT / "forcing-acquisition/data/processed/forcing_hourly_multi.parquet"
    out.to_parquet(p, index=False)
    cov = out.groupby(["stasiun_id", "stasiun_nama"]).agg(start=("waktu", "min"), end=("waktu", "max"), n=("waktu", "size"))
    print(cov.to_string())
    print(f"\nsaved {p}: {len(out)} rows, {out.stasiun_id.nunique()} stations")


if __name__ == "__main__":
    main()
