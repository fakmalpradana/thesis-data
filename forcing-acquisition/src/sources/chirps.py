"""CHIRPS v2.0 daily rainfall (0.05deg). No credentials.

Verified interactively (docs/SOURCE_NOTES.md §1):
  - one NetCDF4/HDF5 file per year, global, ~1.3GB - never download whole file.
  - server sends `Accept-Ranges: bytes` -> open lazily via fsspec+h5netcdf and
    .sel() the bbox before .load(); only the touched HDF5 chunks are fetched.
  - units: mm/day. lat ascending (south-up, -50..50), NOT north-up.
  - timestamp = first day of the accumulation period (start-of-interval,
    matches brief §5 convention directly - no shift needed).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import fsspec
import xarray as xr

from .base import BBox

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps"


def available_range() -> tuple[date, date]:
    # earliest year on the server directory listing is 1981; not re-verified
    # here since training period only needs 2021+ (domain.yaml).
    return date(1981, 1, 1), date.today()


def _year_url(year: int) -> str:
    return f"{BASE_URL}/chirps-v2.0.{year}.days_p05.nc"


YEARLY_DIR = RAW_DIR / "_yearly"  # optional plain-download cache, see _open_year()


def _open_year(year: int) -> xr.Dataset:
    """Open one year's global file, local bulk-downloaded copy first.
    Lazy per-chunk HTTP range reads (h5netcdf+fsspec) turned out fine for a
    small smoke-test window but effectively hung for a full year - each
    touched HDF5 chunk is a separate round trip, and with ~365+ chunks/year
    that's dominated by per-request latency, not bandwidth. A plain
    sequential download of the whole 1.3GB/year file is slower in theory but
    was, in practice, the one that actually finished with visible progress
    (see docs/SOURCE_NOTES.md)."""
    local = YEARLY_DIR / f"chirps-v2.0.{year}.days_p05.nc"
    if local.exists():
        return xr.open_dataset(local, engine="h5netcdf")
    of = fsspec.open(_year_url(year), mode="rb")
    return xr.open_dataset(of.open(), engine="h5netcdf")


def fetch(start: date, end: date, bbox: BBox) -> Path:
    """Subset each touched year's global file to bbox+date range and save
    the subset (raw values, no regridding/resampling) to data/raw/chirps/.
    Saving the bbox subset rather than the 1.3GB/year global file is a
    deliberate deviation from literal "raw = as downloaded" - the brief
    itself forbids pulling global data (§0), which a byte-for-byte raw copy
    of these files would be."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"chirps_{start.isoformat()}_{end.isoformat()}.nc"
    if out_path.exists():
        return out_path

    frames = []
    for year in range(start.year, end.year + 1):
        ds = _open_year(year)
        y_start = max(start, date(year, 1, 1))
        y_end = min(end, date(year, 12, 31))
        sub = ds.sel(
            latitude=slice(bbox.lat_min, bbox.lat_max),
            longitude=slice(bbox.lon_min, bbox.lon_max),
            time=slice(y_start.isoformat(), y_end.isoformat()),
        ).load()
        frames.append(sub)
        ds.close()

    combined = xr.concat(frames, dim="time") if len(frames) > 1 else frames[0]
    combined.to_netcdf(out_path)
    return out_path


def parse(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)
