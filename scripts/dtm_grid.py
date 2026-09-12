"""10 m PINN grid from the 1.5 m DTM: warp, slope/aspect, D8 flow dir/accum, sinks.

Never loads the 1.8 GB DTM at full res in Python - gdalwarp does the resampling,
pysheds only ever touches the 10 m grid (~3771x1979).

    python3 scripts/dtm_grid.py
"""
from pathlib import Path
import subprocess

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DTM = ROOT / "forcing-acquisition/data/static/dem/jakarta/DTM_DKI_HYDRO_JALAN_1.5m.tif"
OUT = ROOT / "forcing-acquisition/data/static/dem/jakarta/grid10m"
GDAL_BIN = "/opt/homebrew/bin"

# bbox lon 106.68-107.02, lat -6.25..-6.07 reprojected to EPSG:32748, rounded
# to a 10 m grid (gdalwarp -te xmin ymin xmax ymax)
TE = (685860, 9308860, 723570, 9328650)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def warp_dtm(dst: Path) -> None:
    run([f"{GDAL_BIN}/gdalwarp", "-overwrite", "-t_srs", "EPSG:32748",
         "-tr", "10", "10", "-te", *map(str, TE), "-r", "average",
         "-dstnodata", "nan", "-ot", "Float32",
         "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
         str(DTM), str(dst)])


def slope_aspect(dtm10: Path) -> None:
    run([f"{GDAL_BIN}/gdaldem", "slope", str(dtm10), str(OUT / "slope_deg.tif"),
         "-compute_edges"])
    run([f"{GDAL_BIN}/gdaldem", "aspect", str(dtm10), str(OUT / "aspect.tif"),
         "-compute_edges"])


def flow_and_sinks(dtm10: Path) -> None:
    # ponytail: pysheds is the one dep the plan allows for D8 flowdir/accum/fill
    from pysheds.grid import Grid

    grid = Grid.from_raster(str(dtm10))
    dem = grid.read_raster(str(dtm10))

    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    fdir = grid.flowdir(inflated)
    facc = grid.accumulation(fdir)

    grid.to_raster(fdir, str(OUT / "fdir.tif"), dtype=np.int32)
    grid.to_raster(facc, str(OUT / "facc.tif"), dtype=np.float32)

    sinks = np.asarray(flooded) - np.asarray(dem)
    sinks[np.asarray(dem.nodata_cells if hasattr(dem, "nodata_cells") else np.isnan(dem))] = np.nan
    with rasterio.open(dtm10) as src:
        profile = src.profile
    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(OUT / "sinks.tif", "w", **profile) as dst:
        dst.write(sinks.astype("float32"), 1)

    return facc, sinks


def print_stats(dtm10: Path, facc, sinks) -> None:
    with rasterio.open(dtm10) as src:
        a = src.read(1)
        cellsize = src.res[0] * src.res[1]
    valid = a[~np.isnan(a)]
    n = valid.size
    flat = np.asarray(rasterio.open(OUT / "slope_deg.tif").read(1))
    flat_valid = flat[~np.isnan(a)]
    pct_flat = float((flat_valid < 0.5).mean() * 100)
    sink_vol = float(np.nansum(np.clip(sinks, 0, None)) * cellsize)

    print(f"valid px: {n}")
    print(f"elevation p1/p50/p99: {np.percentile(valid,1):.2f} / {np.percentile(valid,50):.2f} / {np.percentile(valid,99):.2f} m")
    print(f"% cells slope < 0.5 deg: {pct_flat:.1f}%")
    print(f"max facc: {float(np.nanmax(np.asarray(facc))):.0f} cells")
    print(f"total sink volume: {sink_vol:.0f} m3")

    # self-check: grid shape matches bbox/10 within +-1 px; p50 in plausible orthometric range
    exp_w = round((TE[2] - TE[0]) / 10)
    exp_h = round((TE[3] - TE[1]) / 10)
    assert abs(a.shape[1] - exp_w) <= 1 and abs(a.shape[0] - exp_h) <= 1, "grid shape off bbox/10 - check -te/-tr"
    assert 0 <= np.percentile(valid, 50) <= 6, "p50 elevation outside 0-6 m - datum/grid likely wrong"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dtm10 = OUT / "dtm_10m.tif"
    warp_dtm(dtm10)
    slope_aspect(dtm10)
    facc, sinks = flow_and_sinks(dtm10)
    print_stats(dtm10, facc, sinks)
