"""GSMaP_Gauge v8 hourly rain (0.1deg), FTP, JAXA/EORC. Credentials in ~/.netrc
(curl -n, never printed/logged/argv'd).

Verified interactively 2026-09-12 on 2024-01-15 09Z
(gsmap_gauge.20240115.0900.v8.1000.0.dat.gz):
  - gzip -> float32 little-endian, 3600x3600... no: 3600(lon) x 1200(lat) =
    4,320,000 values, row-major (lat rows x lon cols), units mm/h.
  - grid: lon = 0.05 + i*0.1 for i in 0..3599 (0.05E..359.95E); lat = 59.95 -
    j*0.1 for j in 0..1199 (north->south, 59.95N row 0 down to -59.95 row
    1199). Confirmed on the real file: rows 660..664 (computed from the lat
    formula for -6.0..-6.5) hold values -6.05..-6.45 - matches the doc's
    north-up claim, no flip needed.
  - no negative fill value seen in the sample (min 0.0); doc mentions
    -99/-999 elsewhere in the record so still guard for negatives -> NaN.
  - server is FTP-only; HTTP/HTTPS on this host time out (checked once).
"""
from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from subprocess import run

import numpy as np
import pandas as pd

try:
    from .base import BBox
except ImportError:  # run directly: `python3 forcing-acquisition/src/sources/gsmap.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sources.base import BBox

HOST = "hokusai.eorc.jaxa.jp"
NLON, NLAT = 3600, 1200
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "gsmap"


def _url(dt: datetime, rev: str = "1000") -> str:
    # ponytail: file revision differs by period (early Nov 2023 = v8.0000.0, later = v8.1000.0);
    # _fetch_hour tries 1000 then 0000. Add revisions here if a 550 shows up again.
    return (f"ftp://{HOST}/standard/v8/hourly_G/{dt:%Y/%m/%d}/"
            f"gsmap_gauge.{dt:%Y%m%d.%H}00.v8.{rev}.0.dat.gz")


def available_range() -> tuple[date, date]:
    # per plan: available through 2026-09-09; not re-verified beyond that.
    return date(2000, 3, 1), date(2026, 9, 9)


def lat_row(lat: float) -> int:
    return int(round((59.95 - lat) / 0.1))


def lon_col(lon: float) -> int:
    return int(round((lon - 0.05) / 0.1))


def bbox_slice(bbox: BBox) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    """Row/col slices for bbox, plus the cell-center lat/lon arrays. Computed
    from the grid definition (not hardcoded), asserted to fall inside bbox."""
    r0, r1 = lat_row(bbox.lat_max), lat_row(bbox.lat_min)  # lat_max is more north -> smaller row
    c0, c1 = lon_col(bbox.lon_min), lon_col(bbox.lon_max)
    lat_c = 59.95 - np.arange(r0, r1 + 1) * 0.1
    lon_c = 0.05 + np.arange(c0, c1 + 1) * 0.1
    assert lat_c.min() >= bbox.lat_min - 1e-9 and lat_c.max() <= bbox.lat_max + 1e-9
    assert lon_c.min() >= bbox.lon_min - 1e-9 and lon_c.max() <= bbox.lon_max + 1e-9
    return slice(r0, r1 + 1), slice(c0, c1 + 1), lat_c, lon_c


def _fetch_hour(dt: datetime) -> np.ndarray | None:
    """Download+gunzip one hour, return the (1200,3600) grid or None on failure."""
    tmp = RAW_DIR / f"_tmp_{dt:%Y%m%d%H}.dat.gz"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for rev in ("1000", "0000"):
        r = run(["curl", "-n", "-sS", "--retry", "3", "--max-time", "60", "-o", str(tmp), _url(dt, rev)],
                capture_output=True, text=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            break
        tmp.unlink(missing_ok=True)
    else:
        return None
    try:
        raw = gzip.decompress(tmp.read_bytes())
        arr = np.frombuffer(raw, dtype="<f4").reshape(NLAT, NLON).copy()
    except (OSError, ValueError):
        arr = None
    finally:
        tmp.unlink(missing_ok=True)
    if arr is not None:
        arr[arr < 0] = np.nan
    return arr


def fetch(start: date, end: date, bbox: BBox, workers: int = 6) -> Path:
    """One npz per day under data/raw/gsmap/YYYY/, 24 hourly bbox slices
    (key 'rain' shape (24,nlat,nlon), 'lat', 'lon', 'time'). Skips days whose
    npz already exists (resumable). Missing/failed hours -> NaN slice, hour
    logged to data/raw/gsmap/missing.txt."""
    r_sl, c_sl, lat_c, lon_c = bbox_slice(bbox)
    day = start
    while day <= end:
        out_dir = RAW_DIR / f"{day:%Y}"
        out_path = out_dir / f"gsmap_gauge_{day:%Y%m%d}.npz"
        if out_path.exists():
            day += timedelta(days=1)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        hours = [datetime(day.year, day.month, day.day, h) for h in range(24)]
        slices = {h: None for h in range(24)}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_h = {ex.submit(_fetch_hour, dt): dt.hour for dt in hours}
            for fut in as_completed(fut_to_h):
                h = fut_to_h[fut]
                grid = fut.result()
                slices[h] = grid[r_sl, c_sl] if grid is not None else None
        missing = [h for h, s in slices.items() if s is None]
        if missing:
            with open(RAW_DIR / "missing.txt", "a") as f:
                for h in missing:
                    f.write(f"{day:%Y-%m-%d} {h:02d}00\n")
        nan_slice = np.full((len(lat_c), len(lon_c)), np.nan, dtype="float32")
        rain = np.stack([slices[h] if slices[h] is not None else nan_slice for h in range(24)])
        np.savez(out_path, rain=rain, lat=lat_c, lon=lon_c,
                 time=np.array([f"{day:%Y-%m-%d}T{h:02d}:00:00" for h in range(24)], dtype="datetime64[s]"))
        day += timedelta(days=1)
    return RAW_DIR


def parse(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def to_hourly_series(bbox_npz_dir: Path, points: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Nearest 0.1deg cell per station -> DataFrame[waktu, stasiun_id, hujan_mm]
    (+ a 'bbox_mean' column, same for every station, for reference)."""
    npz_files = sorted(bbox_npz_dir.rglob("gsmap_gauge_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no gsmap npz under {bbox_npz_dir}")
    rows = []
    for p in npz_files:
        d = np.load(p)
        rain, lat, lon, time = d["rain"], d["lat"], d["lon"], d["time"]
        bbox_mean = np.nanmean(rain, axis=(1, 2))
        idx = {sid: (int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin()))
               for sid, (la, lo) in points.items()}
        for sid, (ri, ci) in idx.items():
            rows.append(pd.DataFrame(dict(
                waktu=pd.to_datetime(time), stasiun_id=sid,
                hujan_mm=rain[:, ri, ci].astype(float), bbox_mean=bbox_mean.astype(float),
            )))
    return pd.concat(rows, ignore_index=True).sort_values(["stasiun_id", "waktu"])


def _self_check() -> None:
    """Fetch one real hour (2024-01-15 09Z) via ~/.netrc and check grid shape,
    bbox slicing, and value range. Needs network; that's the point (it's the
    thing this module does)."""
    arr = _fetch_hour(datetime(2024, 1, 15, 9))
    if arr is None:
        print("gsmap._self_check: FTP fetch failed (network/credentials?), skipping")
        return
    assert arr.shape == (1200, 3600), arr.shape
    bbox = BBox(lon_min=106.6, lat_min=-6.5, lon_max=107.1, lat_max=-6.0)
    r_sl, c_sl, lat_c, lon_c = bbox_slice(bbox)
    sl = arr[r_sl, c_sl]
    assert 4 <= sl.shape[0] <= 6 and 4 <= sl.shape[1] <= 6, sl.shape
    nanmax = np.nanmax(sl) if np.isfinite(sl).any() else 0.0
    assert 0 <= nanmax <= 200, nanmax
    print(f"gsmap._self_check OK: shape {arr.shape}, bbox slice {sl.shape}, "
          f"lat centers {lat_c}, nanmax {nanmax:.3f} mm/h")


if __name__ == "__main__":
    _self_check()
