# Forcing Acquisition — PINN-LSTM Banjir Jakarta Utara

Pelengkap [`../tinggi_air`](../tinggi_air) (target). Lihat
[`docs/SOURCE_NOTES.md`](docs/SOURCE_NOTES.md) untuk bukti verify-before-loop
tiap sumber.

## Status (2026-09-07)

| Sumber | Status |
|---|---|
| CHIRPS | ✅ selesai, teruji end-to-end tanpa kredensial |
| Pasut Tanjung Priok (observasi IOC) | 🛑 blocked, lihat SOURCE_NOTES §2 |
| Pasut — EOT20 (model astronomis, keputusan pengguna) | ✅ selesai, teruji (`method="nearest"` untuk titik pesisir) |
| OSM Overpass (drainase) | ✅ selesai, teruji end-to-end tanpa kredensial |
| SoilGrids (WCS) | 🔍 akses terverifikasi, CRS kustom perlu ditangani, belum diimplementasikan |
| ESA WorldCover | 🔍 akses terverifikasi, butuh 2 tile (S06E105+S09E105), belum diimplementasikan |
| FABDEM | 🔍 mirror tanpa akun ditemukan (HF, via STAC), belum diimplementasikan |
| IMERG / ERA5-Land / GloFAS | ⏸️ belum dikerjakan — `~/.netrc` & `~/.cdsapirc` belum ada |
| harmonize.py / qc.py / pipeline.py | ⏸️ belum ditulis |

## Menjalankan smoke test CHIRPS

```bash
cd forcing-acquisition
python3 -c "
import sys; sys.path.insert(0, 'src')
from datetime import date
from sources import chirps
from sources.base import BBox
p = chirps.fetch(date(2024,1,1), date(2024,1,10), BBox(106.6,-6.5,107.1,-6.0))
print(p)
"
```

## Test

```bash
python3 tests/test_chirps.py
```

## Kredensial

Belum ada `~/.netrc` (NASA Earthdata) atau `~/.cdsapirc` (CDS) di mesin ini.
Modul IMERG/ERA5-Land/GloFAS akan gagal dengan pesan jelas begitu ditulis —
**jangan** buat akun/isi kredensial lewat saya, daftar sendiri lalu taruh
file konfigurasinya di lokasi standar.
