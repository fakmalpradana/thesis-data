"""Build/refresh catalog.duckdb - views over the tracked parquet + a manual
registry of every dataset (including the gitignored raw/static ones), so the
DB is the single lookup point regardless of what's in git. Re-run whenever
data changes; idempotent (DROP + recreate each time).

Open catalog.duckdb from DBeaver (built-in DuckDB driver), `duckdb -ui`, or
point QGIS at the paths listed in the `catalog` table for raster/vector.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent  # data/
DB_PATH = ROOT / "catalog.duckdb"

VIEWS = {
    # name: (glob relative to ROOT, description)
    "tma_point": ("tinggi_air/data/processed/[0-9]*_[0-9][0-9][0-9][0-9].parquet",
                  "TMA titik (raw QC-flagged), semua stasiun/tahun"),
    "tma_hourly": ("tinggi_air/data/processed/*_hourly.parquet",
                   "TMA agregat jam-an (tma_mean/tma_max/qc_flag/pct_flagged)"),
    "forcing_hourly": ("forcing-acquisition/data/processed/forcing_hourly.parquet",
                        "TMA+hujan+pasut ter-align jam-an, node 140 (MVP)"),
    "tide_140": ("forcing-acquisition/data/processed/tide_140_2021_2026.parquet",
                 "Prediksi pasut EOT20, node 140, 2021-2026"),
}

# Explicit registry - includes gitignored raw/static files so the catalog
# stays the single source of truth even for what isn't in git. Extend by
# hand when a new dataset shows up; no auto-scanning magic.
CATALOG_ROWS = [
    # name, path, format, layer_kind, tracked_in_git, source, description
    ("tma_point", "tinggi_air/data/processed/*_[0-9][0-9][0-9][0-9].parquet", "parquet", "tabular", True,
     "poskobanjir.dsdadki.web.id (scraper)", "TMA titik per stasiun/tahun, QC-flagged"),
    ("tma_hourly", "tinggi_air/data/processed/*_hourly.parquet", "parquet", "tabular", True,
     "derived from tma_point", "TMA agregat jam-an"),
    ("tma_raw_text", "tinggi_air/data/raw/{stasiun_id}/*.txt", "txt", "tabular", False,
     "poskobanjir.dsdadki.web.id (scraper)", "Respons scraper mentah per rentang tanggal, 14 stasiun"),
    ("forcing_hourly", "forcing-acquisition/data/processed/forcing_hourly.parquet", "parquet", "tabular", True,
     "harmonize.py (TMA+CHIRPS+EOT20)", "Forcing+label ter-align jam-an, node 140"),
    ("tide_140", "forcing-acquisition/data/processed/tide_140_2021_2026.parquet", "parquet", "tabular", True,
     "pyTMD + EOT20", "Prediksi pasut node 140"),
    ("chirps_yearly", "forcing-acquisition/data/raw/chirps/_yearly/chirps-v2.0.{tahun}.days_p05.nc", "netcdf",
     "raster", False, "data.chc.ucsb.edu CHIRPS-2.0 global_daily p05", "Hujan harian global, 2021-2026, per tahun"),
    ("chirps_bbox_subset", "forcing-acquisition/data/raw/chirps/chirps_2021-01-01_2026-09-07.nc", "netcdf",
     "raster", False, "subset lokal dari chirps_yearly", "Hujan harian, bbox Jakarta Utara"),
    ("eot20_constituents", "forcing-acquisition/data/static/tide/EOT20/ocean_tides/{konstituen}_ocean_eot20.nc",
     "netcdf", "raster", False, "DGFI-TUM EOT20 (SEANOE DOI 10.17882/79489)",
     "17 konstituen pasut global, grid 1/8deg"),
    ("demnas", "DEM/DEMNAS_merged_UTM.tif", "geotiff", "raster", False,
     "BIG (user-supplied)", "DEMNAS merged, UTM"),
    ("dtm_jakut_150cm", "forcing-acquisition/data/static/dem/jakut_dtm_150cm/DTM_JAKUT_2023_150cm.tif",
     "geotiff", "raster", False, "user-supplied (symlink to DEM/)", "DTM Jakarta Utara 2023, 1.5m"),
    ("osm_drainage", "forcing-acquisition/data/raw/osm_drainage/osm_drainage_106.6_-6.5_107.1_-6.0.json",
     "geojson", "vector", False, "OSM Overpass API", "Drainase/kanal, 4110 elemen"),
    ("batas_kota_dki", "reference/batas_adm/Batas Kota DKI.geojson", "geojson", "vector", True,
     "reference (sumber tidak tercatat)", "Batas administrasi Kota DKI Jakarta"),
    ("kecamatan_dki", "reference/batas_adm/Kecamatan DKI.geojson", "geojson", "vector", True,
     "reference (sumber tidak tercatat)", "Batas administrasi kecamatan DKI"),
    ("grid_25ha", "reference/jakarta/GRID_25HA.gpkg", "gpkg", "vector", True,
     "reference (analisis internal)", "Grid analisis 25 ha, Jakarta"),
]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    for name, (glob, desc) in VIEWS.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{ROOT / glob}')")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"view {name}: {n} rows - {desc}")

    con.execute("""
        CREATE OR REPLACE TABLE catalog (
            name VARCHAR, path VARCHAR, format VARCHAR, layer_kind VARCHAR,
            tracked_in_git BOOLEAN, source VARCHAR, description VARCHAR
        )
    """)
    con.executemany("INSERT INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?)", CATALOG_ROWS)
    n_cat = con.execute("SELECT count(*) FROM catalog").fetchone()[0]
    print(f"catalog table: {n_cat} entries")

    con.close()
    print(f"\nsaved {DB_PATH}")


if __name__ == "__main__":
    main()
