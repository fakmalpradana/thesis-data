"""Visual explorer for every dataset in catalog.duckdb.

    marimo edit notebooks/explore.py

Each section: overview (source, APA 7 citation, brief analysis) -> interactive view.
Tabular -> table + time series; raster -> decimated imshow (1.8GB DTM opens in
seconds); vector -> geopandas plot.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


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
def _(mo):
    mo.md(
        """
        # Thesis data explorer — PINN-LSTM flood forecasting, Jakarta Utara

        Every dataset registered in `catalog.duckdb`, one section each. Each section opens with
        an **overview** (what it is, where it comes from, APA 7 citation, brief analysis) followed
        by an **interactive view**. Raw data lives outside git; the catalog records its provenance.

        Study target: water level (TMA, *tinggi muka air*) at DKI Jakarta floodgates/pumps, with
        rainfall (CHIRPS) and astronomical tide (EOT20) as physical forcings. MVP node = station
        140, P.A. Marina Ancol (Laut), the only station with a continuous 2021–2026 record.
        """
    )
    return


@app.cell
def _(catalog, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Catalog

                Manual registry (`scripts/build_catalog.py`) of 14 datasets across two pipelines.
                `tracked_in_git = false` rows are large raw/static files kept locally only.
                Tabular datasets are exposed as DuckDB views over parquet; raster/vector rows
                hold file paths for QGIS or this notebook.
                """
            ),
            mo.ui.table(catalog, selection=None),
        ]
    )
    return


@app.cell
def _(con, mo):
    _cov = con.sql(
        """SELECT stasiun_id, stasiun_nama, min(waktu)::date AS start, max(waktu)::date AS "end",
                  count(*) AS hours, round(avg(pct_flagged), 2) AS mean_pct_flagged
           FROM tma_hourly GROUP BY 1, 2 ORDER BY 1"""
    ).df()
    stasiun = (_cov.stasiun_id + " - " + _cov.stasiun_nama).tolist()
    pick_stasiun = mo.ui.dropdown(stasiun, value=next(s for s in stasiun if s.startswith("140")), label="station")
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Water level (TMA) per station — *label / target*

                **Source.** Scraped from the DKI Jakarta Water Resources Agency flood-post portal
                (`poskobanjir.dsdadki.web.id`), 14 floodgate (*pintu air*) and pump stations.
                Raw point observations are irregular (≈2.5–11 min, station- and time-dependent);
                `tma_hourly` aggregates to hourly mean/max with a QC flag and % of flagged points.

                > Dinas Sumber Daya Air Provinsi DKI Jakarta. (2026). *Posko Banjir: Data tinggi
                > muka air pintu air dan pompa* [Data set]. https://poskobanjir.dsdadki.web.id

                **Analysis.** Coverage is very uneven: only station 140 spans 2021–2026
                (~49.8k hourly rows); most others start Nov/Dec 2023 (~24k rows); 114/122/163
                start Aug 2026 (~900 rows) and are unusable for training. Flag rates are low
                (<0.5 %) everywhere, but station 140 shows visible sensor degradation from 2024
                onward (see plot) — the train/val/test split must account for this. Sensor type
                differs (pressure at *pintu air* vs bubble gauges at pumps), so cross-station
                pooling needs per-station normalisation.
                """
            ),
            mo.ui.table(_cov, selection=None),
            pick_stasiun,
        ]
    )
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
def _(con, mo):
    forcing = con.sql("SELECT * FROM forcing_hourly ORDER BY waktu").df()
    _c = forcing[["tma_mean", "hujan_mm", "tide_m"]].corr().round(3)
    rng = mo.ui.date_range(
        start=forcing.waktu.min().date(), stop=forcing.waktu.max().date(),
        value=(forcing.waktu.min().date(), forcing.waktu.max().date()), label="date range",
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. Harmonised forcing table, node 140 — *model input*

                **Source.** `forcing-acquisition/src/harmonize.py` aligns three series to a common
                hourly UTC axis: TMA station 140 (Sec. 2), CHIRPS daily rainfall at the station's
                grid cell (Sec. 4, held constant within each day — `hujan_resolusi = 'harian'`),
                and EOT20 tide prediction (Sec. 5). {len(forcing):,} hourly rows,
                {forcing.waktu.min().date()} → {forcing.waktu.max().date()}.

                **Analysis.** Pearson correlation on the full record:
                TMA–tide **{_c.loc['tma_mean', 'tide_m']}**, TMA–rain **{_c.loc['tma_mean', 'hujan_mm']}**.
                Station 140 sits at the sea outlet, so tide dominates the signal — a PINN loss
                built on the tidal/hydraulic balance is well-motivated here. The weak rain
                correlation is expected at hourly lag 0 with daily-resolution rainfall; lagged
                and upstream-catchment features are needed before rainfall carries predictive
                weight. Sub-daily rainfall (e.g. IMERG) is the main upgrade path.
                """
            ),
            rng,
        ]
    )
    return forcing, rng


@app.cell
def _(forcing, mo, plt, rng):
    _f = forcing[(forcing.waktu.dt.date >= rng.value[0]) & (forcing.waktu.dt.date <= rng.value[1])]
    _fig, _ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _ax[0].plot(_f.waktu, _f.tma_mean, lw=0.5); _ax[0].set_ylabel("TMA (cm)")
    _ax[1].bar(_f.waktu, _f.hujan_mm, width=0.04, color="tab:blue"); _ax[1].set_ylabel("rain (mm/day)")
    _ax[2].plot(_f.waktu, _f.tide_m, lw=0.5, color="tab:green"); _ax[2].set_ylabel("tide (m)")
    mo.vstack([_fig, mo.ui.table(_f, page_size=15)])
    return


@app.cell
def _(ROOT, mo, xr):
    chirps = xr.open_dataset(ROOT / "forcing-acquisition/data/raw/chirps/chirps_2021-01-01_2026-09-07.nc")
    _m = chirps.precip.mean(["latitude", "longitude"])
    _monthly = _m.groupby("time.month").mean().values
    day = mo.ui.slider(0, chirps.sizes["time"] - 1, value=0, label="day index", full_width=True)
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. CHIRPS v2.0 daily rainfall — *forcing*

                **Source.** Climate Hazards Center, UC Santa Barbara; global daily, 0.05°
                (~5.5 km), satellite IR + station blend. Downloaded per year from
                `data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/` and subset to
                the Jakarta bbox (106.6–107.1 E, 6.5–6.0 S): {chirps.sizes['latitude']}×{chirps.sizes['longitude']}
                cells × {chirps.sizes['time']:,} days ({str(chirps.time.values[0])[:10]} → {str(chirps.time.values[-1])[:10]}).

                > Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S.,
                > Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015). The
                > climate hazards infrared precipitation with stations—A new environmental record
                > for monitoring extremes. *Scientific Data, 2*, 150066.
                > https://doi.org/10.1038/sdata.2015.66

                **Analysis.** Clear monsoon seasonality: bbox-mean rainfall peaks in
                Jan–Feb ({_monthly[0]:.1f}/{_monthly[1]:.1f} mm/day) and bottoms in Aug–Sep
                ({_monthly[7]:.1f}/{_monthly[8]:.1f} mm/day); daily bbox max {float(_m.max()):.0f} mm.
                Limitations for urban flood work: 5 km cells smear convective cells across
                Jakarta Utara, and daily totals lose the hourly intensity that drives drainage
                response. CHIRPS' final product lags ~1 month; the series here ends
                {str(chirps.time.values[-1])[:10]}, so the last weeks of TMA have no rain forcing.
                """
            ),
            day,
        ]
    )
    return chirps, day


@app.cell
def _(chirps, day, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    _d = chirps.precip.isel(time=day.value)
    _d.plot(ax=_ax[0], cmap="Blues", vmin=0, vmax=100)
    _ax[0].set_title(str(_d.time.values)[:10])
    chirps.precip.mean(["latitude", "longitude"]).plot(ax=_ax[1], lw=0.5)
    _ax[1].axvline(_d.time.values, color="red")
    _ax[1].set_title("bbox-mean daily rainfall")
    _fig
    return


@app.cell
def _(con, mo, plt):
    tide = con.sql("SELECT * FROM tide_140 ORDER BY 1").df()
    _t = tide.iloc[:, 1]
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. EOT20 astronomical tide, node 140 — *forcing*

                **Source.** Predicted with pyTMD from the EOT20 global empirical ocean tide model
                (DGFI-TUM), 1/8° grid, 17 constituents, at the coordinates of station 140.
                No in-situ tide gauge for Tanjung Priok is publicly available (IOC/GESLA checked,
                none current), so the model prediction stands in for observed sea level.

                > Hart-Davis, M. G., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M., &
                > Seitz, F. (2021). EOT20: A global ocean tide model from multi-mission satellite
                > altimetry. *Earth System Science Data, 13*(8), 3869–3884.
                > https://doi.org/10.5194/essd-13-3869-2021
                >
                > Hart-Davis, M. G., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M., &
                > Seitz, F. (2021). *EOT20 – A global empirical ocean tide model from multi-mission
                > satellite altimetry* [Data set]. SEANOE. https://doi.org/10.17882/79489
                >
                > Sutterley, T. C. (2024). *pyTMD: Python-based tidal prediction software*
                > [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.5555395

                **Analysis.** Range {_t.min():.2f} → {_t.max():.2f} m: Jakarta Bay is microtidal,
                mixed mainly-diurnal. This captures the astronomical component only — no storm
                surge, wind set-up, or sea-level trend/subsidence — so residual TMA − tide at
                station 140 is the meteorological + hydrological signal the model must learn.
                Coastal 1/8° cells can be partly land-masked; the interpolation point was
                verified to fall on a valid ocean cell.
                """
            ),
        ]
    )
    return (tide,)


@app.cell
def _(mo, plt, tide):
    _fig, _ax = plt.subplots(2, 1, figsize=(14, 6))
    _ax[0].plot(tide.iloc[:, 0], tide.iloc[:, 1], lw=0.3); _ax[0].set_title("full record")
    _s = tide.iloc[:24 * 30]
    _ax[1].plot(_s.iloc[:, 0], _s.iloc[:, 1]); _ax[1].set_title("first 30 days (spring–neap cycle)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.ui.table(tide, page_size=10)])
    return


@app.cell
def _(catalog, mo):
    _r = catalog[catalog.format == "geotiff"]
    pick_raster = mo.ui.dropdown(dict(zip(_r.name, _r.path)), value=_r.name.iloc[0], label="raster")
    factor = mo.ui.slider(4, 64, value=16, step=4, label="downsample factor")
    mo.vstack(
        [
            mo.md(
                """
                ## 6. Terrain — DEMNAS and DKI 1.5 m DTM — *physics / static*

                **Sources.**
                - `demnas` — national seamless DEM (BIG), ~8 m (0.27″) from IFSAR/TerraSAR-X/ALOS
                  PALSAR, merged and reprojected to UTM 48S (EPSG:32748). Surface model: includes
                  buildings/vegetation.
                - `dtm_dki_150cm` — 1.5 m bare-earth DTM covering all of DKI Jakarta,
                  hydro-enforced with road surfaces integrated; EPSG:32748, nodata −32767.
                  *Provenance to confirm* — user-supplied from the 2025 DKI Jakarta 3D city
                  mapping programme; cite the official product once the publisher/year is fixed.

                > Badan Informasi Geospasial. (2018). *DEMNAS: Seamless Digital Elevation Model
                > Nasional* [Data set]. https://tanahair.indonesia.go.id/demnas
                >
                > [DTM 1.5 m — citation pending: publisher, year, product title]

                **Analysis.** Jakarta Utara is a low-lying coastal plain: most of the DTM
                histogram sits at 0–5 m, with polder areas at or below sea level — this is the
                geometric reason tide leaks into TMA. DEMNAS is too coarse and too "surface"
                for drainage routing but useful as an upstream-catchment context layer; the
                1.5 m DTM is the appropriate source for slope, flow direction, and depression
                storage terms in the physics-informed loss. The view below is a decimated read
                (1/N) for speed — use QGIS for full resolution.
                """
            ),
            mo.hstack([pick_raster, factor]),
        ]
    )
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
    _fig.colorbar(_im, ax=_ax[0], label="elevation (m)")
    _ax[0].set_title(f"{pick_raster.selected_key} (1/{factor.value})")
    _ax[1].hist(_a[np.isfinite(_a)].ravel(), bins=100)
    _ax[1].set_title("elevation histogram")
    mo.vstack([mo.md(f"`{_info}`"), _fig])
    return


@app.cell
def _(ROOT, catalog, gpd, mo, plt):
    _v = catalog[catalog.layer_kind == "vector"]
    _fig, _ax = plt.subplots(figsize=(10, 9))
    _counts = {}
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
        _counts[_row["name"]] = len(_g)
        _g.plot(ax=_ax, facecolor="none" if _g.geom_type.iloc[0].endswith("Polygon") else None,
                edgecolor=None if not _g.geom_type.iloc[0].endswith("Polygon") else "k",
                lw=0.4, label=f"{_row['name']} ({len(_g)})")
    _ax.legend(fontsize=8)
    _ax.set_title("Vector layers (EPSG:4326)")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 7. Vector layers — drainage network and administrative reference

                **Sources.**
                - `osm_drainage` — OpenStreetMap ways tagged `waterway=drain|canal|river|stream|ditch`
                  via the Overpass API, bbox 106.6–107.1 E / 6.5–6.0 S ({_counts.get('osm_drainage', '?')} ways).
                - `batas_kota_dki`, `kecamatan_dki` — DKI Jakarta city and sub-district boundaries.
                  *Source not recorded in the reference folder* — likely BIG/BPS administrative
                  boundaries; confirm before citing.
                - `grid_25ha` — 25 ha analysis grid, internal product for aggregating results.

                > OpenStreetMap contributors. (2026). *OpenStreetMap* [Data set]. Retrieved
                > September 7, 2026, via Overpass API, from https://www.openstreetmap.org
                >
                > [Administrative boundaries — citation pending: publisher, year]

                **Analysis.** OSM drainage is dense along the 13 main rivers and the Banjir
                Kanal Barat/Timur but sparse for tertiary drains, and geometry carries no
                cross-section, invert, or flow-direction attributes. It is adequate as a
                connectivity graph (which floodgates are hydraulically linked) and as a mask
                for the DTM hydro-enforcement, not as a hydraulic model input. Administrative
                polygons serve only for aggregation/reporting.
                """
            ),
            _fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
