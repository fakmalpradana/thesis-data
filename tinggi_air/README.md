# Akuisisi TMA Pintu Air DKI Jakarta

Pipeline akuisisi otomatis Tinggi Muka Air (TMA) dari portal peringatan dini
banjir DSDA DKI Jakarta, untuk variabel target model PINN-LSTM peramalan
banjir (Jakarta Utara). Lihat [`docs/DISCOVERY.md`](docs/DISCOVERY.md) untuk
temuan discovery (endpoint, batas rentang, kedalaman arsip) dan brief asli
untuk konteks lengkap.

## Cara pakai

```bash
pip install requests pandas pyarrow pyyaml
```

Tarik data (default: rentang di `config/settings.yaml`, semua stasiun):

```bash
python3 src/pipeline.py
```

Stasiun tertentu, rentang tertentu:

```bash
python3 src/pipeline.py --stations 122,114,163 --start 01/01/2025 --end 31/12/2025
```

Mode incremental (lanjut dari timestamp terakhir tersimpan per stasiun):

```bash
python3 src/pipeline.py --incremental
```

Proses **aman dimatikan di tengah** (Ctrl-C) — jalankan ulang perintah yang
sama, chunk yang sudah tersimpan di `data/raw/` tidak ditarik ulang (lihat
komentar di `src/pipeline.py`).

## Struktur output

- `data/raw/{station_id}/{start}_{end}.txt` — respons mentah apa adanya,
  jadi acuan idempotensi/resume sekaligus cadangan kalau parser perlu direvisi.
- `data/processed/{station_id}_{year}.parquet` — data 10-menitan + `qc_flag`.
- `data/processed/{station_id}_{year}_hourly.parquet` — agregasi jam-an
  (`tma_mean`, `tma_max`, `n_obs`, `qc_flag`; flag `9` = gap, `n_obs < 2`).
- `reports/run_YYYYMMDD.md` — ringkasan run: record per stasiun, missing rate
  per bulan, distribusi `qc_flag`, gap terpanjang.
- `config/stations.yaml` — daftar 39 pintu air; `ambang_siaga_cm` terisi
  otomatis saat pull pertama tiap stasiun (ambangnya ikut gratis di respons
  data, lihat DISCOVERY.md §1).

## Testing

```bash
python3 tests/test_parser.py
python3 tests/test_qc.py
```

Pakai fixture tersimpan di `tests/fixtures/`, tidak menyentuh jaringan.

## Contoh cron harian (TIDAK dipasang otomatis - lihat brief §4.5)

```cron
0 2 * * * cd /path/to/tinggi_air && python3 src/pipeline.py --incremental >> logs/cron.log 2>&1
```

## Catatan penting

Datum papan duga tiap pos tidak dipublikasikan dan berbeda-beda — angka
`tma_cm` antar stasiun **tidak bisa dibandingkan langsung** tanpa informasi
offset dari Dinas SDA (lihat brief §7). Deret waktu ini cukup untuk LSTM
per-node, belum cukup untuk model spasial terkopel.
