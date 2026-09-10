"""Visual explorer for every dataset in catalog.duckdb.

    marimo edit notebooks/explore.py

Tabular -> interactive table + time series; raster -> decimated imshow
(reads at reduced resolution so 1.8GB DTM opens in seconds); vector -> geopandas plot.
"""
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import duckdb
    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    import xarray as xr

    ROOT = Path(__file__).resolve().parent.parent
    con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
    catalog = con.sql("SELECT * FROM catalog").df()
    return ROOT, catalog, con, gpd, mo, np, plt, rasterio, xr


@app.cell
def _(catalog, mo):
    mo.vstack([mo.md("# Katalog"), mo.ui.table(catalog, selection=None)])
    return


@app.cell
def _(con, mo):
    mo.md("# TMA per stasiun")
    stasiun = con.sql(
        "SELECT DISTINCT stasiun_id || ' - ' || stasiun_nama AS s FROM tma_hourly ORDER BY 1"
    ).df()["s"].tolist()
    pick_stasiun = mo.ui.dropdown(stasiun, value=next(s for s in stasiun if s.startswith("140")), label="stasiun")
    pick_stasiun
    return (pick_stasiun,)


@app.cell
def _(con, mo, pick_stasiun, plt):
    _sid = pick_stasiun.value.split(" - ")[0]
    tma = con.sql(f"SELECT * FROM tma_hourly WHERE stasiun_id='{_sid}' ORDER BY waktu").df()
    _fig, _ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    _ax[0].plot(tma.waktu, tma.tma_mean, lw=0.4)
    _ax[0].set_ylabel("tma_mean (cm)")
    _ax[1].plot(tma.waktu, tma.pct_flagged, lw=0.4, color="tab:red")
    _ax[1].set_ylabel("% flagged")
    _fig.suptitle(pick_stasiun.value)
    mo.vstack([_fig, mo.ui.table(tma, page_size=15)])
    return


@app.cell
def _(con, mo, plt):
    forcing = con.sql("SELECT * FROM forcing_hourly ORDER BY waktu").df()
    rng = mo.ui.date_range(
        start=forcing.waktu.min().date(), stop=forcing.waktu.max().date(),
        value=(forcing.waktu.min().date(), forcing.waktu.max().date()), label="rentang",
    )
    mo.vstack([mo.md("# Forcing node 140 (TMA + hujan + pasut)"), rng])
    return forcing, rng


@app.cell
def _(forcing, mo, plt, rng):
    _f = forcing[(forcing.waktu.dt.date >= rng.value[0]) & (forcing.waktu.dt.date <= rng.value[1])]
    _fig, _ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _ax[0].plot(_f.waktu, _f.tma_mean, lw=0.5); _ax[0].set_ylabel("TMA (cm)")
    _ax[1].bar(_f.waktu, _f.hujan_mm, width=0.04, color="tab:blue"); _ax[1].set_ylabel("hujan (mm)")
    _ax[2].plot(_f.waktu, _f.tide_m, lw=0.5, color="tab:green"); _ax[2].set_ylabel("pasut (m)")
    mo.vstack([_fig, mo.ui.table(_f, page_size=15)])
    return


@app.cell
def _(ROOT, mo, xr):
    chirps = xr.open_dataset(ROOT / "forcing-acquisition/data/raw/chirps/chirps_2021-01-01_2026-09-07.nc")
    day = mo.ui.slider(0, chirps.sizes["time"] - 1, value=0, label="hari (index)", full_width=True)
    mo.vstack([mo.md("# CHIRPS hujan harian (bbox Jakut)"), day])
    return chirps, day


@app.cell
def _(chirps, day, mo, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    _d = chirps.precip.isel(time=day.value)
    _d.plot(ax=_ax[0], cmap="Blues", vmin=0, vmax=100)
    _ax[0].set_title(str(_d.time.values)[:10])
    chirps.precip.mean(["latitude", "longitude"]).plot(ax=_ax[1], lw=0.5)
    _ax[1].axvline(_d.time.values, color="red")
    _ax[1].set_title("rata-rata bbox harian")
    _fig
    return


@app.cell
def _(catalog, mo):
    _r = catalog[catalog.format == "geotiff"]
    pick_raster = mo.ui.dropdown(dict(zip(_r.name, _r.path)), value=_r.name.iloc[0], label="raster")
    factor = mo.ui.slider(4, 64, value=16, step=4, label="downsample factor")
    mo.vstack([mo.md("# DEM / DTM"), mo.hstack([pick_raster, factor])])
    return factor, pick_raster


@app.cell
def _(ROOT, factor, mo, np, pick_raster, plt, rasterio):
    # ponytail: decimated read (no overviews); build overviews with gdaladdo if this gets slow
    with rasterio.open(ROOT / pick_raster.value) as _src:
        _shape = (_src.height // factor.value, _src.width // factor.value)
        _a = _src.read(1, out_shape=_shape, resampling=rasterio.enums.Resampling.average).astype("float32")
        if _src.nodata is not None:
            _a[_a == _src.nodata] = np.nan
        _ext = [_src.bounds.left, _src.bounds.right, _src.bounds.bottom, _src.bounds.top]
        _info = f"{_src.width}x{_src.height} px, {_src.crs}, res {_src.res[0]:.2f} m, nodata {_src.nodata}"
    _fig, _ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    _im = _ax[0].imshow(_a, extent=_ext, cmap="terrain", vmin=np.nanpercentile(_a, 1), vmax=np.nanpercentile(_a, 99))
    _fig.colorbar(_im, ax=_ax[0], label="elevasi (m)")
    _ax[0].set_title(f"{pick_raster.selected_key} (1/{factor.value})")
    _ax[1].hist(_a[np.isfinite(_a)].ravel(), bins=100)
    _ax[1].set_title("histogram elevasi")
    mo.vstack([mo.md(f"`{_info}`"), _fig])
    return


@app.cell
def _(ROOT, catalog, gpd, mo, plt):
    _v = catalog[catalog.layer_kind == "vector"]
    _fig, _ax = plt.subplots(figsize=(10, 9))
    for _, _row in _v.iterrows():
        _p = ROOT / _row.path
        if not _p.exists():
            continue
        if _row.format == "geojson" and "osm" in _row["name"]:  # raw Overpass JSON, not GeoJSON
            import json
            from shapely.geometry import LineString
            _els = [e for e in json.load(open(_p))["elements"] if e["type"] == "way" and len(e.get("geometry", [])) > 1]
            _g = gpd.GeoDataFrame(
                {"waterway": [e["tags"].get("waterway") for e in _els]},
                geometry=[LineString([(pt["lon"], pt["lat"]) for pt in e["geometry"]]) for e in _els], crs=4326)
        else:
            _g = gpd.read_file(_p).to_crs(4326)
        _g.plot(ax=_ax, facecolor="none" if _g.geom_type.iloc[0].endswith("Polygon") else None,
                edgecolor=None if not _g.geom_type.iloc[0].endswith("Polygon") else "k",
                lw=0.4, label=f"{_row['name']} ({len(_g)})")
    _ax.legend(fontsize=8)
    _ax.set_title("Layer vektor (EPSG:4326)")
    mo.vstack([mo.md("# Vektor"), _fig])
    return


if __name__ == "__main__":
    app.run()
