# Source Notes — Forcing Acquisition

Tanggal discovery: 2026-09-07. Semua request pakai jeda ≥4 detik dan
`User-Agent: ThesisResearchBot/0.1 (+contact: fakmalpradana@gmail.com; academic use, UGM thesis)`.

## 1. CHIRPS v2.0 — SELESAI, end-to-end teruji

- Direktori: `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/`
  — satu file NetCDF4/HDF5 **per tahun**, **global** (seluruh dunia), ~1.3GB/tahun.
  Tidak boleh diunduh utuh (brief §0 melarang data global).
- Server mengirim `Accept-Ranges: bytes` dan file berformat HDF5 (bukan
  NetCDF3 klasik) → dibuka lambat (`fsspec` + `h5netcdf`), lalu `.sel()` bbox
  sebelum `.load()`. Hanya chunk HDF5 yang tersentuh yang benar-benar diunduh
  (smoke test 10 hari × bbox Jakarta = 28KB tersimpan, bukan 1.3GB).
- **Satuan**: `mm/day` (bukan mm/jam — CHIRPS ini produk harian).
- **Orientasi lintang**: ascending / **south-up** (-49.97 → 49.97). Bukan
  north-up. Wajib diperhatikan saat harmonisasi/regrid.
- **Konvensi timestamp**: atribut `comments` eksplisit menyatakan
  *"time variable denotes the first day of the given day"* → sudah
  **start-of-interval**, cocok langsung dengan konvensi brief §5, tidak perlu
  pergeseran.
- NoData: piksel pesisir/tepi mengembalikan `NaN` (bukan sentinel numerik) —
  lihat contoh di smoke test, beberapa sel di garis pantai Jakarta Utara NaN.
- Kedalaman arsip: direktori berisi tahun 2017–2026 minimal (tidak dicek
  mundur lebih jauh — training period 2021+ sudah cukup, lihat domain.yaml).
- Implementasi: [`src/sources/chirps.py`](../src/sources/chirps.py).

**Update setelah production pull 2021–2026 (bukan sekadar smoke test 10
hari):** pendekatan lazy byte-range di atas ternyata **tidak scale** — untuk
1 tahun penuh, jumlah HDF5 chunk yang tersentuh (~365+, kemungkinan 1
chunk/hari) membuat prosesnya didominasi latensi per-request, bukan
bandwidth. Dites >30 menit tanpa kemajuan (CPU idle, menunggu jaringan)
untuk rentang 2021–2026, dan bahkan >2 menit untuk 1 tahun saja dengan
block_size diperbesar. **Diganti** jadi unduh utuh per-file-tahun secara
sekuensial biasa (`curl`, bukan byte-range) ke `data/raw/chirps/_yearly/`,
baru di-subset bbox secara lokal — lebih lambat secara teori (unduh 1.3GB
penuh) tapi progresnya terlihat dan **benar-benar selesai**, tidak seperti
pendekatan lazy yang macet. `chirps.py` sekarang mengecek file lokal ini
dulu sebelum jatuh ke lazy-fetch (dipertahankan untuk kasus smoke-test kecil
yang memang terbukti cepat).

## 2. Pasang surut — IOC Tanjung Priok — **BLOCKED, brief §10 stop-condition terpicu**

Brief eksplisit: *"Kalau langkah 3 menunjukkan data pasut Tanjung Priok tidak
memadai untuk periode training, berhenti dan laporkan sebelum lanjut."*
Itulah yang terjadi.

- **Tidak ada stasiun bernama "Tanjung Priok"** di jaringan IOC Sea Level
  Monitoring saat ini (`list.php?showall=all`, 1768 baris, semua stasiun
  Indonesia diperiksa manual).
- Stasiun terdekat secara geografis: **`koli` — "Kolinamil, Jakarta Port"**
  (Kolinamil = pelabuhan militer di kompleks Tanjung Priok). Kondisi:
  - Update terakhir: **2026-08-10 11:32** — mati **~28 hari** per tanggal
    discovery (2026-09-07). Bacaan terakhir bernilai sentinel `-999`.
  - `koli2` ("Kolinamil 2"): mati sejak **2025-07-23** (~411 hari) — praktis
    tidak aktif.
  - Endpoint `bgraph.php?code=koli&output=tab` **hanya melayani jendela live
    pendek** (~30 hari terakhir dari server, bukan arsip). Dicoba
    `startdate=`/`enddate=` mundur ke 2020, 2023, 2025, dan bahkan ke Agustus
    2026 yang *seharusnya* ada datanya (dari query default) — semuanya
    mengembalikan **0 baris**. Param historis untuk endpoint ini kemungkinan
    beda nama/mekanisme dari yang didokumentasikan secara umum, atau memang
    IOC live monitoring tidak menyediakan bulk-historis via endpoint ini sama
    sekali (arsip resminya ada di pihak ketiga, lihat GESLA-3 di bawah).
  - Field yang tersedia saat live: `pr2(m)`, `prs(m)`, `rad`, `bat` — belum
    diverifikasi mana yang merupakan "sea level" siap pakai vs bacaan sensor
    mentah (biasanya `prs` = pressure-derived sea level, `rad` = radar) —
    tidak diverifikasi lebih jauh karena stasiunnya sendiri sudah tidak layak.
- **GESLA-3**: situs `gesla.org` tidak dapat diakses dari environment ini
  (connection failure/000, kemungkinan diblokir jaringan sandbox, bukan
  situsnya down — perlu dicoba ulang dari environment lain). **Belum
  terverifikasi** apakah GESLA-3 punya stasiun Jakarta/Tanjung Priok sama
  sekali — arsip GESLA umumnya bersumber dari jaringan PSMSL/UHSLC yang
  historisnya tipis untuk Indonesia.
- **FES2022/TPXO** (model pasut astronomis) belum dicoba — ini fallback yang
  masih berlaku terlepas dari temuan di atas, tapi brief sendiri menandaskan
  hasilnya **tidak memuat storm surge**, harus dinyatakan eksplisit sebagai
  keterbatasan di tesis kalau dipakai sebagai pengganti observasi.

**Kesimpulan sementara**: observasi pasut real-time Tanjung Priok/Jakarta
lewat IOC **tidak memadai** untuk periode training manapun yang berarti —
bahkan 28 hari terakhir pun kosong di titik pull ini. Ini mengubah desain
eksperimen (brief §10), bukan detail implementasi — **berhenti di sini,
menunggu keputusan pengguna** sebelum menulis modul `ioc_tide.py` penuh.
Opsi yang tersedia dibahas di chat, bukan diputuskan sepihak di sini.

## 2b. Pasang surut — keputusan pengguna: FES2022/TPXO (model, bukan observasi)

Pengguna memilih opsi "model pasut saja" setelah temuan §2. Verifikasi akses:

- **FES2022** (AVISO+): **butuh akun** (approval bisa berhari-hari) — sama
  seperti CDS/Earthdata, di luar aturan §3 (tidak boleh saya buatkan akun).
- **TPXO**: juga butuh registrasi/lisensi OSU untuk resolusi tinggi.
- **EOT20** (DGFI-TUM, via SEANOE, DOI 10.17882/79489): **CC BY 4.0, tanpa
  akun**. Diverifikasi: halaman dataset dapat diakses, link unduh langsung
  `https://www.seanoe.org/data/00683/79489/data/85762.zip` (~2GB, seluruh
  dunia, resolusi 1/8°, semua konstituen pasut utama, format NetCDF per
  konstituen di dalam zip). Server **tidak** mendukung byte-range (dites
  `-r 0-10485759`, server mengabaikan dan mengirim body penuh) — tidak bisa
  subset sebagian file lewat HTTP, jadi ini **satu kali unduh utuh**, bukan
  time series berulang. Sejalan dengan semangat brief §1.4 (lapisan
  statis/model, sekali unduh) meskipun secara struktur dokumen ada di §1.3.
  **Keterbatasan yang harus dinyatakan di tesis**: EOT20/FES/TPXO adalah
  prediksi pasut **astronomis murni — tidak memuat storm surge**.
- **Selesai.** Unduhan (2.3GB zip berisi 2 sub-zip: `load_tides.zip` tidak
  dipakai, `ocean_tides.zip` 1.2GB yang dipakai) diekstrak ke
  `data/static/tide/EOT20/ocean_tides/` (17 file NetCDF, satu per konstituen
  pasut, ~130MB masing-masing, format persis sesuai konvensi penamaan yang
  diharapkan pyTMD `{const}_ocean_eot20.nc`). Zip asli dihapus setelah
  ekstraksi (hemat ~2.3GB, isinya sudah redundan dengan hasil ekstrak).
- **Temuan orientasi penting (§2 wajib)**: file konstituen mentah menyimpan
  **lon 0°–360°**, bukan -180°..180°. `pyTMD.compute.tide_elevations`
  menangani wrap ini secara internal (param `crs=4326`), tapi kalau modul
  ditulis manual (baca netCDF langsung tanpa pyTMD) ini wajib dikonversi.
- **Temuan interpolasi**: `method="linear"` pyTMD mengembalikan **NaN** di
  titik dekat pantai Teluk Jakarta — grid 1/8° (~14km) di sana punya sel
  darat (NaN) bersebelahan langsung dengan sel laut, dan interpolasi linear
  ke titik pecahan ikut merata-ratakan tetangga NaN. **Keputusan**: pakai
  `method="nearest"` untuk titik pesisir/teluk — trade-off tidak
  menginterpolasi dalam grid basah, dianggap dapat diterima untuk satu titik
  forcing di resolusi 1/8°.
- Satuan hasil: meter (SI langsung, tidak perlu konversi).
- Smoke test nyata (Teluk Jakarta, dekat Tanjung Priok, 106.875°E -6.0°S,
  48 jam): amplitudo pasut ±0.3an meter — masuk akal untuk Teluk Jakarta yang
  mikrotidal. Implementasi: [`src/sources/tide.py`](../src/sources/tide.py).

## 3. Lapisan statis — verifikasi akses (belum diimplementasikan penuh)

- **OSM Overpass** — ✅ selesai & teruji. `POST` query Overpass QL biasa ke
  `overpass-api.de/api/interpreter`, JSON, tanpa auth. Bbox domain penuh
  (106.6,-6.5,107.1,-6.0) → 4110 elemen waterway/embankment, ~30 detik.
  Implementasi: [`src/sources/osm_drainage.py`](../src/sources/osm_drainage.py).
- **SoilGrids (WCS)** — dapat diakses tanpa auth (`DescribeCoverage` berhasil
  untuk `sand_0-5cm_mean`), TAPI **CRS native-nya bukan EPSG standar** —
  `EPSG:152160` (Homolosine kustom ISRIC), satuan meter, bukan derajat.
  `GetCoverage` butuh bbox dalam CRS itu (reprojeksi definisi CRS dulu, WKT
  kustom ISRIC) — belum dikerjakan, dicatat sebagai langkah berikutnya.
- **ESA WorldCover** — tile S3 dapat diakses tanpa auth (dites
  `S06E105_Map.tif`, 200 OK). **Catatan penting**: bbox domain (lat -6.5 s/d
  -6.0) melintasi batas dua tile grid 3°×3° (S06 mencakup -6..-3, S09
  mencakup -9..-6) — perlu **dua tile** (`S06E105` dan `S09E105`), bukan satu,
  lalu di-mosaic. Belum diimplementasikan.
- **FABDEM** — mirror publik tanpa akun ditemukan di Hugging Face
  (`huggingface.co/datasets/links-ads/fabdem-v12`), lisensi CC BY-NC-SA 4.0.
  Root berisi katalog STAC (`stac_catalog/`, `collection.json`) bukan tile
  langsung — perlu ditelusuri lewat STAC untuk dapat URL tile Jakarta. Belum
  diimplementasikan.

## 4. Sumber ber-kredensial — belum dikerjakan (urutan brief §10)

- **IMERG** (`earthaccess`) dan **ERA5-Land/GloFAS** (`cdsapi`): `~/.netrc`
  dan `~/.cdsapirc` **belum ada** di mesin ini (dicek `test -f`, bukan dibaca
  isinya — brief §3). Modul-modulnya akan gagal dengan pesan jelas + cara
  daftar, bukan diam-diam skip, begitu ditulis. Belum ditulis karena tanpa
  kredensial tidak bisa dilakukan smoke test sungguhan (§2 aturan wajib),
  dan menulis loop dari ingatan tanpa verifikasi melanggar aturan itu sendiri.
- **Lapisan statis** (WorldCover, OSM Overpass, SoilGrids, FABDEM): belum
  dikerjakan — tidak butuh kredensial, giliran berikutnya setelah keputusan
  pasut di atas.
