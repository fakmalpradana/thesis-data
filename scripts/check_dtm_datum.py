"""Is the 1.5 m DTM ellipsoidal or orthometric? Compare against DEMNAS (orthometric,
EGM2008 per BIG). Ellipsoidal heights in Jakarta would sit ~+18 m above DEMNAS; an
orthometric DTM sits within a few metres (bare-earth vs coarse DSM). Run:
    python3 scripts/check_dtm_datum.py
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
DTM = ROOT / "forcing-acquisition/data/static/dem/jakarta/DTM_DKI_HYDRO_JALAN_1.5m.tif"
DEMNAS = ROOT / "forcing-acquisition/data/static/dem/demnas/DEMNAS_merged_UTM.tif"
BBOX = (106.70, -6.20, 107.00, -6.08)  # Jakut, lon/lat
N_JAKARTA = 18.0  # EGM2008 geoid undulation, m (pyproj EPSG:4979->9518 at Ancol = 17.999)


def median_diff() -> float:
    t = Transformer.from_crs("EPSG:4326", "EPSG:32748", always_xy=True)
    x0, y0 = t.transform(BBOX[0], BBOX[1]); x1, y1 = t.transform(BBOX[2], BBOX[3])
    out = []
    for p in (DTM, DEMNAS):
        with rasterio.open(p) as src:
            a = src.read(1, window=src.window(x0, y0, x1, y1), out_shape=(300, 600)).astype("float32")
            a[a == src.nodata] = np.nan
            out.append(a)
    return float(np.nanmedian(out[0] - out[1]))


if __name__ == "__main__":
    d = median_diff()
    verdict = "ELLIPSOIDAL (~+N above DEMNAS)" if abs(d - N_JAKARTA) < 5 else "ORTHOMETRIC (~0 vs DEMNAS)"
    print(f"median(DTM - DEMNAS) over Jakut = {d:+.2f} m  ->  {verdict}")
    assert abs(d) < 5, "DTM no longer looks orthometric - re-check datum before using it"
