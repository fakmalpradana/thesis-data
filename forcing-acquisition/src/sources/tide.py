"""Pasang surut - EOT20 (DGFI-TUM, CC BY 4.0, no account). Astronomic tide
prediction only - NO storm surge (state this explicitly in the thesis,
brief §1.3). Chosen over FES2022/TPXO because both of those need an AVISO+ /
OSU account (brief §3 forbids creating one) - IOC live observation for
Tanjung Priok turned out unusable (see docs/SOURCE_NOTES.md §2: nearest
station `koli` has been dead for 28 days at discovery time).

One-time static download (docs/SOURCE_NOTES.md §2b): the SEANOE server
doesn't honor HTTP range requests, so there's no way to fetch less than the
full ~1.2GB `ocean_tides.zip` (17 constituent NetCDF files, one per
constituent, global, 1/8deg). Already extracted to
data/static/tide/EOT20/ocean_tides/ - this module only predicts from it,
it does not re-download.

Verified interactively:
  - raw constituent files store lon as 0..360, not -180..180. pyTMD's
    `compute.tide_elevations` handles the wrap internally (crs=4326).
  - method="linear" returns NaN near the coast: Jakarta Bay's 1/8deg (~14km)
    grid has land-masked NaN cells right next to water ones, and linear
    interpolation to a query point pulls in the NaN neighbor. method="nearest"
    avoids this - used here deliberately, at the cost of not interpolating
    within the wet grid (acceptable at 1/8deg for a single forcing point).
  - output units: meter (already SI, no unit conversion needed).
  - a station's real-world coordinate can land on a **land-masked cell** at
    this 1/8deg resolution even when the station is a working sea-facing
    floodgate (Jakarta's coast is intricate relative to ~14km cells) -
    "nearest grid cell" is not the same as "nearest wet grid cell". Marina
    Ancol (106.8291,-6.1215) itself is land-masked in this grid; the nearest
    wet cell is 106.875,-6.0, ~14km (0.13deg) away. `nearest_wet_point()`
    below searches for it instead of trusting the raw coordinate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyTMD.compute as compute
import xarray as xr

from pathlib import Path

TIDE_DIR = Path(__file__).resolve().parents[2] / "data" / "static" / "tide"
EPOCH = (1992, 1, 1, 0, 0, 0)
_MASK_FILE = TIDE_DIR / "EOT20" / "ocean_tides" / "M2_ocean_eot20.nc"


def nearest_wet_point(lon: float, lat: float, search_deg: float = 1.0) -> tuple[float, float]:
    """Nearest EOT20 grid cell to (lon,lat) that actually has tide data
    (M2 amplitude not NaN) - see module docstring."""
    with xr.open_dataset(_MASK_FILE) as ds:
        sub = ds["amplitude"].sel(
            lon=slice(lon - search_deg, lon + search_deg),
            lat=slice(lat - search_deg, lat + search_deg),
        ).load()
    lons, lats = np.meshgrid(sub.lon.values, sub.lat.values)
    dist2 = (lons - lon) ** 2 + (lats - lat) ** 2
    dist2_wet = np.where(sub.notnull().values, dist2, np.inf)
    if np.isinf(dist2_wet).all():
        raise ValueError(f"no wet EOT20 cell within {search_deg} deg of ({lon},{lat})")
    j, i = np.unravel_index(np.argmin(dist2_wet), dist2_wet.shape)
    return float(lons[j, i]), float(lats[j, i])


def predict(lon: float, lat: float, times: pd.DatetimeIndex) -> pd.Series:
    """Astronomic tide elevation (m) at one point, for the given UTC times."""
    utc = times.tz_convert("UTC") if times.tz is not None else times.tz_localize("UTC")
    delta_time = (utc.values - np.datetime64("1992-01-01")) / np.timedelta64(1, "s")
    da = compute.tide_elevations(
        np.array([lon]), np.array([lat]), delta_time,
        directory=TIDE_DIR, model="EOT20", epoch=EPOCH,
        type="time series", method="nearest",
    )
    return pd.Series(da.values.flatten(), index=times, name="tide_m")


if __name__ == "__main__":
    # Self-check (brief's own tests are fixture-based, but a 17x130MB
    # constituent set isn't fixture-able - this runs against the real local
    # EOT20 data instead, which is the actual dependency being verified).
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="Asia/Jakarta")
    wlon, wlat = nearest_wet_point(106.8291, -6.1215)  # Marina Ancol itself is land-masked
    assert (wlon, wlat) == (106.875, -6.0)
    s = predict(wlon, wlat, idx)
    assert s.notna().all(), "expected no NaN at a known wet grid cell"
    assert s.abs().max() < 2.0, "Jakarta Bay is microtidal, ~2m would be implausible"
    print("ok", s.iloc[:5].to_dict())
