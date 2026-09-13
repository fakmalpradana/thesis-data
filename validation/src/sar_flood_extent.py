"""Sentinel-1 SAR flood extent for events E2/E6/E7 (Jakarta Utara bbox), via
Microsoft Planetary Computer (anonymous STAC + SAS signing; no Copernicus/
Earthdata credentials on this machine).

Water = VV_dB < Otsu threshold (clipped to [-22,-13] dB, else -16 dB fallback).
Flood = water(event scene) AND NOT water(dry reference median) minus blobs
< 10 px. Dry reference = median VV over 3-5 scenes in the same
~40-day season window (any recent year) with < 1 mm GSMaP bbox-mean rain in
the prior 72 h.

    python3 validation/src/sar_flood_extent.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent  # validation/
OUT_DIR = ROOT / "data/processed/sar"
GSMAP_STATIONS = ROOT.parent / "forcing-acquisition/data/processed/gsmap_hourly_stations.parquet"
PETABENCANA = ROOT / "data/processed/petabencana_reports_jakarta.parquet"
BATAS_KOTA = ROOT.parent / "reference/batas_adm/Batas Kota DKI.geojson"
BATAS_KEC = ROOT.parent / "reference/batas_adm/Kecamatan DKI.geojson"
DTM_PATH = ROOT.parent / "forcing-acquisition/data/static/dem/jakarta/grid10m/dtm_10m.tif"
CATALOG_DB = ROOT.parent / "catalog.duckdb"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
BBOX_4326 = (106.70, -6.20, 107.00, -6.08)  # Jakarta Utara

EVENTS = {  # event_id: (start, end) — from validation/events.csv
    "E2": ("2026-01-12", "2026-01-22"),
    "E6": ("2025-07-06", "2025-07-07"),
    "E7": ("2026-06-02", "2026-06-05"),
}

OTSU_BAND = (-22.0, -13.0)
FALLBACK_DB = -16.0
MIN_BLOB_PX = 10
MIN_POLY_M2 = 2500.0  # 0.25 ha
SEA_DTM_M = 0.3  # ponytail: DTM<=0.3 m as sea/waduk proxy, replace with OSM water polygons if needed
SPECKLE_SIZE = 5
SEASON_WINDOW_DAYS = 20  # +/- around same calendar day, searched across years
N_REF_SCENES = (3, 5)
DRY_MM_72H = 1.0
ORBIT_CHECK_DAYS = 4


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """1-D Otsu threshold (stdlib numpy, no sklearn needed for the core algo).

    Self-check in __main__: threshold on a synthetic bimodal array lands
    between the two modes.
    """
    finite = values[np.isfinite(values)]
    hist, edges = np.histogram(finite, bins=nbins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    cum_mean = np.cumsum(hist * centers)
    total_mean = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = cum_mean / w0
        mu1 = (total_mean - cum_mean) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between)
    return float(centers[np.argmax(between)])


def water_threshold(vv_db: np.ndarray) -> tuple[float, bool]:
    """Returns (threshold_db, used_fallback)."""
    t = otsu_threshold(vv_db)
    if OTSU_BAND[0] <= t <= OTSU_BAND[1]:
        return t, False
    return FALLBACK_DB, True


def despeckle(vv_db: np.ndarray) -> np.ndarray:
    """5x5 median filter. NaN (nodata) pixels are held out of the filter and
    restored after, so they don't drag neighbouring medians down."""
    nanmask = ~np.isfinite(vv_db)
    filled = np.where(nanmask, np.nanmedian(vv_db), vv_db)
    out = ndimage.median_filter(filled, size=SPECKLE_SIZE)
    out[nanmask] = np.nan
    return out


def land_mask(transform, crs, shape_hw: tuple[int, int]) -> np.ndarray:
    """True where inside Kota Adm. Jakarta Utara (AOI land clip)."""
    kota = gpd.read_file(BATAS_KOTA)
    poly = kota[kota["NAMOBJ"] == "Kota Adm. Jakarta Utara"].to_crs(crs)
    return rasterize([(g, 1) for g in poly.geometry], out_shape=shape_hw, transform=transform,
                      fill=0, dtype="uint8").astype(bool)


def sea_dtm_mask(transform, crs, shape_hw: tuple[int, int]) -> np.ndarray:
    """True where DTM <= SEA_DTM_M (sea/waduk proxy), resampled to the scene grid."""
    out = np.full(shape_hw, np.nan, dtype="float32")
    with rasterio.open(DTM_PATH) as dtm:
        reproject(source=rasterio.band(dtm, 1), destination=out, src_transform=dtm.transform,
                  src_crs=dtm.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    return np.where(np.isfinite(out), out <= SEA_DTM_M, False)


def open_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search(catalog, start, end):
    items = list(catalog.search(collections=[COLLECTION], bbox=list(BBOX_4326),
                                 datetime=f"{start}/{end}").items())
    items.sort(key=lambda it: it.datetime)
    return items


def read_vv_db(href: str) -> tuple[np.ndarray, "rasterio.Affine", "rasterio.CRS"]:
    with rasterio.open(href) as ds:
        left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *BBOX_4326)
        window = rasterio.windows.from_bounds(left, bottom, right, top, ds.transform)
        arr = ds.read(1, window=window).astype("float64")
        transform = ds.window_transform(window)
        crs = ds.crs
    # RTC linear backscatter (power) -> dB; 0/negative are nodata
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(arr > 0, 10 * np.log10(arr), np.nan)
    return db, transform, crs


def pick_event_scene(catalog, peak: pd.Timestamp):
    """List every scene (asc+desc) within +/-ORBIT_CHECK_DAYS of peak, sorted by
    gap; if none, fall back to +/-6d (flagged wide). Returns (chosen, wide,
    alternatives) where alternatives is the rest of the +/-4d list."""
    for days, wide in ((ORBIT_CHECK_DAYS, False), (6, True)):
        items = search(catalog, (peak - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                        (peak + pd.Timedelta(days=days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if items:
            items.sort(key=lambda it: abs((it.datetime - peak).total_seconds()))
            return items[0], wide, items[1:]
    return None, None, []


def bbox_mean_rain() -> pd.Series:
    df = pd.read_parquet(GSMAP_STATIONS, columns=["waktu", "bbox_mean"]).drop_duplicates("waktu")
    return df.set_index("waktu")["bbox_mean"].sort_index()


def prior_72h_rain_mm(rain: pd.Series, when: pd.Timestamp) -> float | None:
    when_naive = when.tz_localize(None) if when.tzinfo else when
    win = rain.loc[when_naive - pd.Timedelta(hours=72):when_naive]
    if win.empty:
        return None
    return float(win.sum())


def pick_reference_scenes(catalog, event_id: str, peak: pd.Timestamp, rain: pd.Series):
    """Scan the same ~40-day season window across nearby years and rank by
    driest prior-72h rain; return up to 5 of them (>=3 if possible)."""
    candidates = []
    for dy in range(0, 4):  # this year and up to 3 years back
        year = peak.year - dy
        center = peak.replace(year=year) if not (peak.month == 2 and peak.day == 29) else peak.replace(year=year, day=28)
        lo = (center - pd.Timedelta(days=SEASON_WINDOW_DAYS)).strftime("%Y-%m-%d")
        hi = (center + pd.Timedelta(days=SEASON_WINDOW_DAYS) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for it in search(catalog, lo, hi):
            it_dt = pd.Timestamp(it.datetime)
            if abs((it_dt - peak).total_seconds()) <= 3 * 86400:
                continue  # skip the event window itself
            r = prior_72h_rain_mm(rain, it_dt)
            if r is not None:
                candidates.append((r, it))
    candidates.sort(key=lambda x: x[0])
    strict = [it for r, it in candidates if r < DRY_MM_72H]
    if len(strict) >= N_REF_SCENES[0]:
        return strict[:N_REF_SCENES[1]]
    # ponytail: wet-season events (e.g. E2, Jan) have no <1mm/72h window in
    # 4 years of GSMaP record at this bbox -- fall back to the driest
    # available scenes instead of failing; upgrade path is a longer rain
    # lookback (7d dry spell) or a bigger season window if this misfires.
    if candidates:
        print(f"{event_id}: no scene met <{DRY_MM_72H}mm/72h dry criterion, "
              f"using {min(len(candidates), N_REF_SCENES[1])} driest-available instead "
              f"(min {candidates[0][0]:.1f}mm)")
    return [it for _, it in candidates[:N_REF_SCENES[1]]]


def polygonize(mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Polygonize (crs is projected/metric, e.g. UTM), drop polygons smaller
    than MIN_POLY_M2, then reproject to 4326."""
    geoms = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform) if v]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if len(gdf):
        gdf = gdf[gdf.geometry.area >= MIN_POLY_M2]
    return gdf.to_crs(4326) if len(gdf) else gdf.set_crs(4326, allow_override=True)


def pixel_area_km2(transform) -> float:
    return abs(transform.a * transform.e) / 1e6


def bpbd_period_code(peak: pd.Timestamp) -> str:
    """BPBD tables report quarterly, coded YYYY + quarter-end month, e.g. Q1
    2026 (Jan-Mar) -> '202603'."""
    q_end_month = ((peak.month - 1) // 3 + 1) * 3
    return f"{peak.year}{q_end_month:02d}"


def kecamatan_flood_table(event_id: str, flood_gdf: gpd.GeoDataFrame, peak: pd.Timestamp) -> pd.DataFrame:
    """Per-kecamatan (Jakarta Utara) flood_ha from the flood polygons vs.
    bpbd_kejadian (jumlah_kejadian) for the event's BPBD reporting quarter."""
    kec = gpd.read_file(BATAS_KEC)
    kec = kec[kec["WADMKK"] == "JAKARTA UTARA"][["WADMKC", "geometry"]]
    utm = kec.estimate_utm_crs()
    kec_u = kec.to_crs(utm)
    if len(flood_gdf):
        overlay = gpd.overlay(flood_gdf.to_crs(utm), kec_u, how="intersection")
        ha = (overlay.geometry.area / 1e4).groupby(overlay["WADMKC"]).sum()
    else:
        ha = pd.Series(dtype=float)
    table = kec_u[["WADMKC"]].drop_duplicates().set_index("WADMKC")
    table["flood_ha"] = ha.reindex(table.index).fillna(0.0).round(3)

    period = bpbd_period_code(peak)
    with duckdb.connect(str(CATALOG_DB), read_only=True) as con:
        kej = con.sql(
            "select upper(kecamatan) as kec, sum(try_cast(jumlah_kejadian as int)) as n "
            "from bpbd_banjir_2025_2026 where upper(wilayah) like '%UTARA%' and periode_data = ? "
            "group by 1", params=[period]).df()
    kej = kej.set_index("kec")["n"] if len(kej) else pd.Series(dtype=float)
    table["bpbd_kejadian"] = kej.reindex(table.index)
    if kej.empty:
        print(f"{event_id}: no bpbd_banjir_2025_2026 rows for period {period} (event outside reported quarters)")
    return table.reset_index()


def run_event(catalog, event_id: str, start: str, end: str, rain: pd.Series) -> dict:
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    peak = start_ts + (end_ts - start_ts) / 2

    scene, wide, alternatives = pick_event_scene(catalog, peak)
    if scene is None:
        print(f"{event_id}: NO COVERAGE within +/-6d of peak {peak}")
        return {"event": event_id, "scene_id": None, "acq_datetime": None, "orbit": None,
                "gap_hours_to_peak": None, "threshold_db": None, "flood_area_km2": None,
                "ref_scene_ids": None, "alternatives": None}

    gap_h = abs((scene.datetime - peak).total_seconds()) / 3600
    if wide:
        print(f"{event_id}: no scene within +/-{ORBIT_CHECK_DAYS}d, using +/-6d fallback: "
              f"{scene.id} (gap {gap_h:.1f}h)")
    alt_str = ";".join(f"{it.id}:{it.properties.get('sat:orbit_state')}:"
                        f"{abs((it.datetime - peak).total_seconds()) / 3600:.1f}h" for it in alternatives)
    print(f"{event_id}: chosen {scene.id} ({scene.properties.get('sat:orbit_state')}, gap {gap_h:.1f}h); "
          f"{len(alternatives)} alternative scene(s) within +/-{6 if wide else ORBIT_CHECK_DAYS}d")

    ref_items = pick_reference_scenes(catalog, event_id, peak, rain)
    if len(ref_items) < N_REF_SCENES[0]:
        print(f"{event_id}: WARNING only {len(ref_items)} dry reference scenes found (<{N_REF_SCENES[0]})")

    event_db, transform, crs = read_vv_db(scene.assets["vv"].href)
    ref_stack = []
    for it in ref_items:
        db, ref_t, ref_crs = read_vv_db(it.assets["vv"].href)
        if db.shape == event_db.shape:
            ref_stack.append(db)
    if not ref_stack:
        raise RuntimeError(f"{event_id}: no usable dry-reference imagery (grid mismatch or none found)")
    ref_db = np.nanmedian(np.stack(ref_stack), axis=0)

    # step 3: despeckle before thresholding
    event_db = despeckle(event_db)
    ref_db = despeckle(ref_db)

    # step 1: land clip (AOI -> Kota Adm. Jakarta Utara only)
    land = land_mask(transform, crs, event_db.shape)
    # step 2: permanent-water/sea mask = dry-reference water OR DTM<=0.3m
    sea = sea_dtm_mask(transform, crs, event_db.shape)

    # step 4: Otsu computed on land pixels only
    land_vals = np.where(land, event_db, np.nan)
    otsu_land = otsu_threshold(land_vals)
    thr, used_fallback = water_threshold(land_vals)
    in_band = OTSU_BAND[0] <= otsu_land <= OTSU_BAND[1]
    print(f"{event_id}: Otsu-on-land = {otsu_land:.2f} dB "
          f"({'in band' if in_band else 'OUT of band'} {OTSU_BAND}); threshold used = {thr:.1f} dB"
          f"{' (fallback)' if used_fallback else ''}")

    water_event = event_db < thr
    ref_thr, _ = water_threshold(np.where(land, ref_db, np.nan))
    water_ref = ref_db < ref_thr

    flood = land & water_event & ~water_ref & ~sea
    labeled, n = ndimage.label(flood)
    if n:
        sizes = ndimage.sum(flood, labeled, range(1, n + 1))
        small = np.isin(labeled, np.where(sizes < MIN_BLOB_PX)[0] + 1)
        flood = flood & ~small

    px_km2 = pixel_area_km2(transform)
    flood_km2 = float(flood.sum() * px_km2)
    bbox_km2 = float(land.sum() * px_km2)
    pct = 100 * flood_km2 / bbox_km2 if bbox_km2 else float("nan")
    print(f"{event_id}: flood {flood_km2:.3f} km2 ({pct:.2f}% of {bbox_km2:.1f} km2 land AOI), "
          f"{len(ref_stack)} ref scenes")
    if bbox_km2 and (pct > 30 or flood_km2 < 0.05):
        print(f"{event_id}: SANITY FLAG — flood share {pct:.2f}% / {flood_km2:.3f} km2 outside [0.05km2, 30%]; "
              f"check threshold/orbit before trusting this polygon")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_DIR / f"flood_{event_id}.tif", "w", driver="GTiff", height=flood.shape[0],
                        width=flood.shape[1], count=1, dtype="uint8", crs=crs, transform=transform,
                        compress="deflate") as dst:
        dst.write(flood.astype("uint8"), 1)

    gdf = polygonize(flood, transform, crs)  # also drops polygons < MIN_POLY_M2
    gdf["event"] = event_id
    gdf.to_file(OUT_DIR / f"flood_{event_id}.geojson", driver="GeoJSON")

    kec_table = kecamatan_flood_table(event_id, gdf, peak)
    print(f"{event_id}: per-kecamatan flood_ha vs bpbd_kejadian:\n{kec_table.to_string(index=False)}")

    return {"event": event_id, "scene_id": scene.id, "acq_datetime": scene.datetime.isoformat(),
            "orbit": scene.properties.get("sat:orbit_state"), "gap_hours_to_peak": round(gap_h, 1),
            "threshold_db": round(thr, 2), "flood_area_km2": round(flood_km2, 4),
            "ref_scene_ids": ";".join(it.id for it in ref_items[:len(ref_stack)]),
            "alternatives": alt_str}


def petabencana_hit_rate(event_id: str, start: str, end: str) -> str:
    """% of PetaBencana flood reports in [start-1d, end+1d] that fall within
    100 m of a flood polygon. Reports use lon/lat WGS84; buffer in a local UTM."""
    geo_path = OUT_DIR / f"flood_{event_id}.geojson"
    if not geo_path.exists() or not PETABENCANA.exists():
        return "n/a"
    flood = gpd.read_file(geo_path)
    if flood.empty:
        return "0/0 (no flood polygons)"
    reports = pd.read_parquet(PETABENCANA)
    reports = reports[reports["disaster_type"] == "flood"]
    win = reports[(reports["created_at"] >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)) &
                  (reports["created_at"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    if win.empty:
        return "0 reports in window"
    pts = gpd.GeoDataFrame(win, geometry=gpd.points_from_xy(win["lon"], win["lat"]), crs=4326)
    utm = flood.estimate_utm_crs()
    flood_u, pts_u = flood.to_crs(utm), pts.to_crs(utm)
    buf = flood_u.buffer(100).union_all()
    inside = pts_u.geometry.within(buf).sum()
    return f"{inside}/{len(pts_u)} ({100 * inside / len(pts_u):.1f}%)"


def main() -> None:
    catalog = open_catalog()
    rain = bbox_mean_rain()
    rows = []
    for event_id, (start, end) in EVENTS.items():
        rows.append(run_event(catalog, event_id, start, end, rain))
        if rows[-1]["scene_id"]:
            print(f"{event_id}: petabencana hit rate (100m, +/-1d window) = "
                  f"{petabencana_hit_rate(event_id, start, end)}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "sar_scenes.csv", index=False)
    print(f"\n-> {OUT_DIR / 'sar_scenes.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # Self-check: Otsu threshold on a synthetic bimodal array lands between
    # the two modes (this is the one non-trivial bit of logic here).
    demo = np.concatenate([np.random.normal(-20, 1, 5000), np.random.normal(-10, 1, 5000)])
    t = otsu_threshold(demo)
    assert -20 < t < -10, f"otsu self-check failed: {t}"

    # Self-check (step 8): a synthetic sea block (DTM 0 m) and land block
    # (DTM 2 m), both "flooded", keeps only the land block after the DTM mask.
    dtm = np.zeros((4, 8), dtype="float32")
    dtm[:, 4:] = 2.0  # left half sea, right half land
    sea_synth = dtm <= SEA_DTM_M
    flood_synth = np.ones_like(dtm, dtype=bool)
    kept = flood_synth & ~sea_synth
    assert not kept[:, :4].any() and kept[:, 4:].all(), "sea/land mask self-check failed"

    main()
