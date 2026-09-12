"""Fetch GSMaP_Gauge v8 hourly rain (Jakarta Utara bbox) via FTP.

    python3 forcing-acquisition/scripts/fetch_gsmap.py --start 2026-01-01 --end 2026-01-31 [--workers 6]
    python3 forcing-acquisition/scripts/fetch_gsmap.py --build-stations   # skip fetch, just (re)build the parquet

Resumable: re-running the same range skips days whose npz already exists
(see src/sources/gsmap.py fetch()). --build-stations reads ALL npz under
data/raw/gsmap/ (not just --start/--end) and writes
data/processed/gsmap_hourly_stations.parquet - run it by hand once a fetch
covers the range you need (not part of the plain fetch call above, so the
long full-history fetch doesn't also pay for this on every invocation).
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sources.base import BBox  # noqa: E402
from sources.gsmap import fetch, to_hourly_series, RAW_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent  # data/
JAKUT_BBOX = BBox(lon_min=106.6, lat_min=-6.5, lon_max=107.1, lat_max=-6.0)
JAKUT = [107, 126, 140, 150, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187]
STATION_170_FALLBACK = 140  # no coords in stations.yaml -> reuse node-140 cell


def jakut_points() -> dict[int, tuple[float, float]]:
    stations = yaml.safe_load((ROOT / "tinggi_air/config/stations.yaml").read_text())["stations"]
    by_id = {s["id"]: s.get("koordinat") for s in stations if s["id"] in JAKUT}
    points = {sid: (k["lat"], k["lon"]) for sid, k in by_id.items() if k}
    assert STATION_170_FALLBACK in points, "fallback station itself has no coords"
    for sid in JAKUT:
        if by_id.get(sid) is None:
            points[sid] = points[STATION_170_FALLBACK]
    return points


def build_stations() -> Path:
    df = to_hourly_series(RAW_DIR, jakut_points())
    out = ROOT / "forcing-acquisition/data/processed/gsmap_hourly_stations.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out}: {len(df)} rows, {df.stasiun_id.nunique()} stations, "
          f"{df.waktu.min()} -> {df.waktu.max()}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--build-stations", action="store_true")
    args = p.parse_args()
    if not args.build_stations:
        if not (args.start and args.end):
            raise SystemExit("--start/--end required unless --build-stations")
        out = fetch(args.start, args.end, JAKUT_BBOX, workers=args.workers)
        print(f"done: {args.start} -> {args.end}, npz under {out}")
    else:
        build_stations()
