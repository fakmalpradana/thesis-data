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
import yaml

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
    "forcing_hourly_multi": ("forcing-acquisition/data/processed/forcing_hourly_multi.parquet",
                             "TMA 14 stasiun Jakut + hujan + pasut (shared node-140 cell) jam-an"),
    "bpbd_banjir_2025_2026": ("validation/data/processed/data-kejadian-bencana-banjir.parquet",
                              "BPBD kejadian banjir per kelurahan/bulan, 2025Q1-2026Q1"),
    "bpbd_banjir_2024": ("validation/data/processed/data-kejadian-bencana-banjir-tahun-2024.parquet",
                         "BPBD kejadian banjir per kelurahan/bulan, 2024"),
    "bpbd_banjir_2023": ("validation/data/processed/data-kejadian-bencana-banjir-tahun-2023.parquet",
                         "BPBD kejadian banjir per kelurahan/bulan, 2023"),
    "bpbd_banjir_2020": ("validation/data/processed/data-kejadian-bencana-banjir-di-provinsi-dki-jakarta-tahun-2020.parquet",
                         "BPBD kejadian banjir per RW dengan tanggal & lama genangan, 2020"),
    "dsda_pintu_air": ("validation/data/processed/data-pintu-air.parquet", "DSDA pintu air + koordinat, 2024"),
    "dsda_rumah_pompa": ("validation/data/processed/data-lokasi-rumah-pompa.parquet",
                         "DSDA rumah pompa + koordinat + kapasitas, 2024"),
    "petabencana_reports": ("validation/data/processed/petabencana_reports_jakarta.parquet",
                            "PetaBencana.id laporan warga ber-geotag + kedalaman, DKI 2021-2026"),
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
    ("forcing_hourly_multi", "forcing-acquisition/data/processed/forcing_hourly_multi.parquet", "parquet", "tabular", True,
     "harmonize_multi.py", "Forcing+label jam-an, 14 stasiun Jakut, 369k baris"),
    ("tide_140", "forcing-acquisition/data/processed/tide_140_2021_2026.parquet", "parquet", "tabular", True,
     "pyTMD + EOT20", "Prediksi pasut node 140"),
    ("chirps_yearly", "forcing-acquisition/data/raw/chirps/_yearly/chirps-v2.0.{tahun}.days_p05.nc", "netcdf",
     "raster", False, "data.chc.ucsb.edu CHIRPS-2.0 global_daily p05", "Hujan harian global, 2021-2026, per tahun"),
    ("chirps_bbox_subset", "forcing-acquisition/data/raw/chirps/chirps_2021-01-01_2026-09-07.nc", "netcdf",
     "raster", False, "subset lokal dari chirps_yearly", "Hujan harian, bbox Jakarta Utara"),
    ("eot20_constituents", "forcing-acquisition/data/static/tide/EOT20/ocean_tides/{konstituen}_ocean_eot20.nc",
     "netcdf", "raster", False, "DGFI-TUM EOT20 (SEANOE DOI 10.17882/79489)",
     "17 konstituen pasut global, grid 1/8deg"),
    ("demnas", "forcing-acquisition/data/static/dem/demnas/DEMNAS_merged_UTM.tif", "geotiff", "raster", False,
     "BIG (user-supplied)", "DEMNAS merged, UTM"),
    ("dtm_dki_150cm", "forcing-acquisition/data/static/dem/jakarta/DTM_DKI_HYDRO_JALAN_1.5m.tif",
     "geotiff", "raster", False, "user-supplied (DCKTRP DKI)",
     "DTM se-DKI Jakarta (hydro-enforced + jalan), 1.5m. Vertikal: ORTOMETRIK (geoid belum dikonfirmasi; "
     "konsisten dengan DEMNAS EGM2008 +/-1 m, pantai 0.3-1 m). Klaim 'elipsoid WGS 84' dari DCKTRP tidak "
     "sesuai isi raster - selisih thd DEMNAS ~0 m, bukan ~18 m. Cek 2026-09-12, scripts/check_dtm_datum.py"),
    ("osm_drainage", "forcing-acquisition/data/raw/osm_drainage/osm_drainage_106.6_-6.5_107.1_-6.0.json",
     "geojson", "vector", False, "OSM Overpass API", "Drainase/kanal, 4110 elemen"),
    ("batas_kota_dki", "reference/batas_adm/Batas Kota DKI.geojson", "geojson", "vector", True,
     "reference (sumber tidak tercatat)", "Batas administrasi Kota DKI Jakarta"),
    ("kecamatan_dki", "reference/batas_adm/Kecamatan DKI.geojson", "geojson", "vector", True,
     "reference (sumber tidak tercatat)", "Batas administrasi kecamatan DKI"),
    ("grid_25ha", "reference/jakarta/GRID_25HA.gpkg", "gpkg", "vector", True,
     "reference (analisis internal)", "Grid analisis 25 ha, Jakarta"),
    ("bpbd_banjir_*", "validation/data/processed/data-kejadian-bencana-banjir*.parquet", "parquet", "tabular", True,
     "satudata.jakarta.go.id (BPBD DKI) via validation/src/fetch_satudata.py",
     "Kejadian banjir per kelurahan: 2019, 2020 (per RW + tanggal), 2021, 2023, 2024, 2025-2026Q1; ketinggian air, RW/KK terdampak"),
    ("bpbd_rekap_2013_2020", "validation/data/processed/data-rekapitulasi-tahunan-kejadian-banjir-di-provinsi-dki-jakarta.parquet",
     "parquet", "tabular", True, "satudata.jakarta.go.id (BPBD DKI)", "Rekap tahunan banjir per kelurahan 2013-2020"),
    ("luasan_tergenang_2023", "validation/data/processed/luasan-daerah-tergenang-tahun-2023.parquet", "parquet", "tabular", True,
     "satudata.jakarta.go.id (DSDA DKI)", "Luas genangan per RT 2022 vs 2023 (ha)"),
    ("dsda_pintu_air", "validation/data/processed/data-pintu-air.parquet", "parquet", "tabular", True,
     "satudata.jakarta.go.id (DSDA DKI)", "72 pintu air DKI + lat/lon + sistem aliran"),
    ("dsda_rumah_pompa", "validation/data/processed/data-lokasi-rumah-pompa.parquet", "parquet", "tabular", True,
     "satudata.jakarta.go.id (DSDA DKI)", "614 rumah pompa DKI + lat/lon + kapasitas"),
    ("bpbd_titik_rawan", "validation/data/processed/data-titik-rawan-bencanabanjir.parquet", "parquet", "tabular", True,
     "satudata.jakarta.go.id (BPBD DKI)", "154 titik rawan banjir + lat/lon (koordinat tanpa desimal, bagi 1e6)"),
    ("stations", "tinggi_air/config/stations.yaml", "yaml", "tabular", True,
     "poskobanjir (id/nama/ambang siaga) + DSDA satudata (koordinat, by-name match)",
     "39 stasiun TMA; 14 Jakut punya koordinat + ambang Siaga 1/2/3"),
    ("events", "validation/events.csv", "csv", "tabular", True,
     "hand-curated from tma_hourly + bpbd_banjir_* + petabencana_reports",
     "Kandidat kejadian banjir Jakut 2024-2026 untuk validasi (E1-E7), status candidate/confirmed"),
    ("petabencana_reports", "validation/data/processed/petabencana_reports_jakarta.parquet", "parquet", "tabular", True,
     "data.petabencana.id archive API via validation/src/fetch_petabencana.py",
     "19,325 laporan warga DKI 2021-2026 (12,000 banjir) + kedalaman cm + lat/lon"),
]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    # stations table from tinggi_air/config/stations.yaml (id, nama, koordinat DSDA, ambang siaga)
    st = yaml.safe_load((ROOT / "tinggi_air/config/stations.yaml").read_text())["stations"]
    con.execute("""CREATE OR REPLACE TABLE stations (
        stasiun_id INTEGER, nama VARCHAR, lat DOUBLE, lon DOUBLE, koordinat_sumber VARCHAR,
        siaga1_cm DOUBLE, siaga2_cm DOUBLE, siaga3_cm DOUBLE)""")
    con.executemany("INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        (x["id"], x["nama"], (x.get("koordinat") or {}).get("lat"), (x.get("koordinat") or {}).get("lon"),
         (x.get("koordinat") or {}).get("sumber"), x["ambang_siaga_cm"]["siaga1"], x["ambang_siaga_cm"]["siaga2"],
         x["ambang_siaga_cm"]["siaga3"]) for x in st])
    print(f"stations table: {len(st)} rows")
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_csv('{ROOT / 'validation/events.csv'}')")
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
