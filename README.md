# thesis-data

Data acquisition + catalog for a PINN-LSTM Jakarta Utara flood forecasting
thesis. Two pipelines, one catalog:

- **`tinggi_air/`** — TMA (tinggi muka air, water level) scraper, the
  model's label/target. 14 Jakut pintu air/pompa stasiun (140 from 2021,
  the rest from Nov 2023). See
  `tinggi_air/docs/DISCOVERY.md`.
- **`forcing-acquisition/`** — forcing/input data: hujan (CHIRPS), pasut
  (EOT20 tide model), drainase (OSM). See
  `forcing-acquisition/docs/SOURCE_NOTES.md`.

- **`validation/`** — ground truth for evaluation: BPBD flood-event tables
  (2019–2026Q1), DSDA pintu air / rumah pompa with coordinates, PetaBencana
  crowdsourced reports (2021–2026), and `events.csv` — the curated list of
  candidate events. Fetchers: `validation/src/fetch_satudata.py`,
  `validation/src/fetch_petabencana.py`.

## Catalog

`catalog.duckdb` indexes everything — both the processed parquet tracked in
this repo and the large raw/static files that are gitignored (CHIRPS
NetCDF, EOT20 constituents, DEM/DTM rasters). It's the single place to look
up what data exists and where, even for files not in git.

Rebuild after any data change:

```bash
python3 scripts/build_catalog.py
```

**Visual explorer (marimo):** every dataset plotted interactively — TMA per
stasiun, forcing node 140, CHIRPS day slider, DEM/DTM decimated view +
histogram, all vector layers.

```bash
pip install marimo && marimo edit notebooks/explore.py
```

**Browse it:**
- Tabular data (TMA, forcing, tide series): open `catalog.duckdb` in
  [DBeaver](https://dbeaver.io/) (built-in DuckDB driver) or run
  `duckdb -ui catalog.duckdb` for a quick interactive spreadsheet view.
- Raster/vector layers (CHIRPS, EOT20, DEM/DTM, OSM drainage, admin
  boundaries): query `SELECT * FROM catalog WHERE layer_kind IN ('raster',
  'vector')` for paths, then open those paths directly in QGIS.

## Experiments

`experiments/lstm_baseline.py` trains an LSTM baseline for TMA at station 140
and compares 3 train/val/test split configs against a persistence baseline
(`python3 experiments/lstm_baseline.py`). Results: `reports/lstm_baseline/`,
viewer: `marimo edit notebooks/02_lstm_baseline.py`.

`experiments/lstm_multistation.py` repeats the baseline for all 14 Jakut
stations on `forcing_hourly_multi` (split train Nov 2023–Dec 2024 / val
Jan–Apr 2025 / test May 2025–Sep 2026). Results: `reports/lstm_multistation/`.

`experiments/siaga_f1.py` scores Siaga-3 exceedance (F1/CSI/POD/FAR, LSTM vs
persistence) per station/horizon and per event (E2/E6/E7) against BPBD +
PetaBencana ground truth. `notebooks/03_xcorr_events.py` cross-correlates
TMA with tide/rain and classifies each event pluvial/tidal/compound.
Results: `reports/siaga_f1/`, `reports/xcorr/`.

## Raw data

Not tracked in git (see `.gitignore` in each pipeline + repo root) — either
too large (CHIRPS ~7GB, EOT20 ~1.2GB, DEM rasters) or user-supplied. Every
gitignored dataset is still listed in the `catalog` table with its source,
and regenerable via the fetch scripts documented in each pipeline's `docs/`.
