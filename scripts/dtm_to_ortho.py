"""DTM_DKI_HYDRO_JALAN_1.5m.tif (WGS 84 ellipsoidal heights) -> orthometric,
decimated to 10 m. Never loads the 1.8 GB full-res raster into memory - reads
once via a decimated (out_shape) request with average resampling.

INAGeoid2020 v2 (EPSG:20036, DSDA's official vertical datum) grid is not
downloadable without a request to BIG (srgi@big.go.id / 021-8753155) - no
self-serve GTX/GeoTIFF/xyz link on https://srgi.big.go.id. Falls back to
EGM2008 (pyproj network grid, EPSG:9518).
# ponytail: EGM2008 stand-in, swap grid for INAGeoid2020 v2 when BIG provides it.

    python3 scripts/dtm_to_ortho.py

Writes (gitignored, under forcing-acquisition/data/static/dem/jakarta/):
  DTM_DKI_10m_ortho.tif  - orthometric height H, float32, EPSG:32748
  DTM_DKI_10m_N.tif      - geoid separation N used, same grid
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
DEM_DIR = ROOT / "forcing-acquisition/data/static/dem/jakarta"
SRC = DEM_DIR / "DTM_DKI_HYDRO_JALAN_1.5m.tif"
OUT_H = DEM_DIR / "DTM_DKI_10m_ortho.tif"
OUT_N = DEM_DIR / "DTM_DKI_10m_N.tif"
TARGET_RES_M = 10.0


def read_decimated(src_path: Path, target_res: float):
    with rasterio.open(src_path) as src:
        src_res = src.transform.a  # 1.5 m, square pixels
        out_h = max(1, round(src.height * src_res / target_res))
        out_w = max(1, round(src.width * src_res / target_res))
        data = src.read(
            1, out_shape=(out_h, out_w), resampling=Resampling.average,
        ).astype("float32")
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        nodata = src.nodata
        crs = src.crs
    valid = data != nodata
    data = np.where(valid, data, np.nan).astype("float32")
    return data, transform, crs, valid


def geoid_separation(data: np.ndarray, transform, crs, valid: np.ndarray) -> np.ndarray:
    """N(x,y) on the same grid as `data` via EGM2008 (pyproj network grid)."""
    pyproj.network.set_network_enabled(True)
    rows, cols = np.where(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs, ys = np.asarray(xs), np.asarray(ys)

    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_egm = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:9518", always_xy=True)
    lon, lat = to_ll.transform(xs, ys)
    _, _, h0_as_egm = to_egm.transform(lon, lat, np.zeros_like(lon))
    n_valid = -h0_as_egm  # h=0 ellipsoidal -> EGM2008 H = 0 - N, so N = -H

    N = np.full(data.shape, np.nan, dtype="float32")
    N[rows, cols] = n_valid.astype("float32")
    return N


def summary(name: str, arr: np.ndarray) -> None:
    v = arr[~np.isnan(arr)]
    print(f"{name}: min={v.min():.3f} median={np.median(v):.3f} max={v.max():.3f} (n={v.size})")


def write_tif(path: Path, data: np.ndarray, transform, crs) -> None:
    profile = dict(
        driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        tiled=True, blockxsize=256, blockysize=256, compress="LZW",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def main() -> None:
    print(f"reading {SRC} decimated to {TARGET_RES_M} m ...")
    h, transform, crs, valid = read_decimated(SRC, TARGET_RES_M)
    summary("h (ellipsoidal, WGS84)", h)

    print("computing N (EGM2008) on the 10 m grid ...")
    N = geoid_separation(h, transform, crs, valid)
    summary("N (EGM2008 geoid separation)", N)

    H = h - N
    summary("H (orthometric, EGM2008)", H)

    DEM_DIR.mkdir(parents=True, exist_ok=True)
    write_tif(OUT_H, H, transform, crs)
    write_tif(OUT_N, N, transform, crs)
    print(f"wrote {OUT_H}")
    print(f"wrote {OUT_N}")

    # self-check: N at Ancol (approx -6.126, 106.83) should be 17.5-18.5 m (EGM2008, per plan)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ax, ay = to_utm.transform(106.83, -6.126)
    row, col = rasterio.transform.rowcol(transform, ax, ay)
    n_ancol = float(N[row, col])
    assert 17.5 <= n_ancol <= 18.5, f"N at Ancol out of expected range: {n_ancol}"
    print(f"self-check OK: N at Ancol = {n_ancol:.3f} m")


if __name__ == "__main__":
    main()
