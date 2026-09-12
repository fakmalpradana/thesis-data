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

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent  # validation/
OUT_DIR = ROOT / "data/processed/sar"
GSMAP_STATIONS = ROOT.parent / "forcing-acquisition/data/processed/gsmap_hourly_stations.parquet"
PETABENCANA = ROOT / "data/processed/petabencana_reports_jakarta.parquet"

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
SEASON_WINDOW_DAYS = 20  # +/- around same calendar day, searched across years
N_REF_SCENES = (3, 5)
DRY_MM_72H = 1.0


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
    """Nearest scene within +/-3d, else +/-6d (flagged), else None."""
    for days, wide in ((3, False), (6, True)):
        items = search(catalog, (peak - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                        (peak + pd.Timedelta(days=days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if items:
            items.sort(key=lambda it: abs((it.datetime - peak).total_seconds()))
            return items[0], wide
    return None, None


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
    geoms = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform) if v]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf.to_crs(4326) if len(gdf) else gdf.set_crs(4326, allow_override=True)


def pixel_area_km2(transform) -> float:
    return abs(transform.a * transform.e) / 1e6


def run_event(catalog, event_id: str, start: str, end: str, rain: pd.Series) -> dict:
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    peak = start_ts + (end_ts - start_ts) / 2

    scene, wide = pick_event_scene(catalog, peak)
    if scene is None:
        print(f"{event_id}: NO COVERAGE within +/-6d of peak {peak}")
        return {"event": event_id, "scene_id": None, "acq_datetime": None, "orbit": None,
                "gap_hours_to_peak": None, "threshold_db": None, "flood_area_km2": None, "ref_scene_ids": None}

    gap_h = abs((scene.datetime - peak).total_seconds()) / 3600
    if wide:
        print(f"{event_id}: no scene within +/-3d, using +/-6d fallback: {scene.id} (gap {gap_h:.1f}h)")

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

    thr, used_fallback = water_threshold(event_db)
    water_event = event_db < thr
    ref_thr, _ = water_threshold(ref_db)
    water_ref = ref_db < ref_thr

    flood = water_event & ~water_ref
    labeled, n = ndimage.label(flood)
    if n:
        sizes = ndimage.sum(flood, labeled, range(1, n + 1))
        small = np.isin(labeled, np.where(sizes < MIN_BLOB_PX)[0] + 1)
        flood = flood & ~small

    px_km2 = pixel_area_km2(transform)
    flood_km2 = float(flood.sum() * px_km2)
    bbox_km2 = float(np.isfinite(event_db).sum() * px_km2)
    pct = 100 * flood_km2 / bbox_km2 if bbox_km2 else float("nan")
    print(f"{event_id}: flood {flood_km2:.3f} km2 ({pct:.2f}% of {bbox_km2:.1f} km2 bbox), "
          f"threshold {thr:.1f} dB{' (fallback)' if used_fallback else ''}, "
          f"{len(ref_stack)} ref scenes")
    if bbox_km2 and (pct > 30 or flood_km2 < 0.05):
        print(f"{event_id}: SANITY FLAG — flood share {pct:.2f}% / {flood_km2:.3f} km2 outside [0.05km2, 30%]; "
              f"check threshold/orbit before trusting this polygon")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_DIR / f"flood_{event_id}.tif", "w", driver="GTiff", height=flood.shape[0],
                        width=flood.shape[1], count=1, dtype="uint8", crs=crs, transform=transform,
                        compress="deflate") as dst:
        dst.write(flood.astype("uint8"), 1)

    gdf = polygonize(flood, transform, crs)
    gdf["event"] = event_id
    gdf.to_file(OUT_DIR / f"flood_{event_id}.geojson", driver="GeoJSON")

    return {"event": event_id, "scene_id": scene.id, "acq_datetime": scene.datetime.isoformat(),
            "orbit": scene.properties.get("sat:orbit_state"), "gap_hours_to_peak": round(gap_h, 1),
            "threshold_db": round(thr, 2), "flood_area_km2": round(flood_km2, 4),
            "ref_scene_ids": ";".join(it.id for it in ref_items[:len(ref_stack)])}


def petabencana_hit_rate(event_id: str, start: str, end: str) -> str:
    """% of PetaBencana reports in the event window that fall inside the flood
    polygon (50m buffer). Reports use lon/lat WGS84; buffer in a local UTM."""
    geo_path = OUT_DIR / f"flood_{event_id}.geojson"
    if not geo_path.exists() or not PETABENCANA.exists():
        return "n/a"
    flood = gpd.read_file(geo_path)
    if flood.empty:
        return "0/0 (no flood polygons)"
    reports = pd.read_parquet(PETABENCANA)
    win = reports[(reports["created_at"] >= pd.Timestamp(start, tz="UTC")) &
                  (reports["created_at"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    if win.empty:
        return "0 reports in window"
    pts = gpd.GeoDataFrame(win, geometry=gpd.points_from_xy(win["lon"], win["lat"]), crs=4326)
    utm = flood.estimate_utm_crs()
    flood_u, pts_u = flood.to_crs(utm), pts.to_crs(utm)
    buf = flood_u.buffer(50).union_all()
    inside = pts_u.geometry.within(buf).sum()
    return f"{inside}/{len(pts_u)} ({100 * inside / len(pts_u):.1f}%)"


def main() -> None:
    catalog = open_catalog()
    rain = bbox_mean_rain()
    rows = []
    for event_id, (start, end) in EVENTS.items():
        rows.append(run_event(catalog, event_id, start, end, rain))
        if rows[-1]["scene_id"]:
            print(f"{event_id}: petabencana hit rate = {petabencana_hit_rate(event_id, start, end)}")

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
    main()
