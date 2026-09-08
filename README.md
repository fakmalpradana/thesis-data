# thesis-data

Data acquisition + catalog for a PINN-LSTM Jakarta Utara flood forecasting
thesis. Two pipelines, one catalog:

- **`tinggi_air/`** — TMA (tinggi muka air, water level) scraper, the
  model's label/target. 11 pintu air stasiun, 2021-2026. See
  `tinggi_air/docs/DISCOVERY.md`.
- **`forcing-acquisition/`** — forcing/input data: hujan (CHIRPS), pasut
  (EOT20 tide model), drainase (OSM). See
  `forcing-acquisition/docs/SOURCE_NOTES.md`.

## Catalog

`catalog.duckdb` indexes everything — both the processed parquet tracked in
this repo and the large raw/static files that are gitignored (CHIRPS
NetCDF, EOT20 constituents, DEM/DTM rasters). It's the single place to look
up what data exists and where, even for files not in git.

Rebuild after any data change:

```bash
python3 scripts/build_catalog.py
```

**Browse it:**
- Tabular data (TMA, forcing, tide series): open `catalog.duckdb` in
  [DBeaver](https://dbeaver.io/) (built-in DuckDB driver) or run
  `duckdb -ui catalog.duckdb` for a quick interactive spreadsheet view.
- Raster/vector layers (CHIRPS, EOT20, DEM/DTM, OSM drainage, admin
  boundaries): query `SELECT * FROM catalog WHERE layer_kind IN ('raster',
  'vector')` for paths, then open those paths directly in QGIS.

## Raw data

Not tracked in git (see `.gitignore` in each pipeline + repo root) — either
too large (CHIRPS ~7GB, EOT20 ~1.2GB, DEM rasters) or user-supplied. Every
gitignored dataset is still listed in the `catalog` table with its source,
and regenerable via the fetch scripts documented in each pipeline's `docs/`.
