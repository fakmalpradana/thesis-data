# Discovery — Portal TMA DSDA DKI Jakarta

Tanggal discovery: 2026-09-07. Semua request memakai jeda ≥4 detik dan
`User-Agent: ThesisResearchBot/0.1 (+contact: fakmalpradana@gmail.com; academic use, UGM thesis)`.

## 1. Mekanisme transport

**Bukan VIEWSTATE.** Halaman `grafikDataTinggiMukaAir.aspx` memang ASP.NET
WebForms dengan `__VIEWSTATE` dkk, tapi tombol "Lihat" **tidak** melakukan
postback penuh. Dari Network tab, klik "Lihat" memicu satu GET sederhana ke:

```
GET /Pages/GenerateDataTinggiAir.aspx?IdPintuAir={id}&StartDate={dd/MM/yyyy}&EndDate={dd/MM/yyyy}
```

Tanpa header/cookie session khusus, tanpa VIEWSTATE. Endpoint ini cukup untuk
seluruh pipeline — pendekatan VIEWSTATE di `scrape_tma_dsdadki.py` (kerangka
awal) dibuang sepenuhnya.

### Contoh respons mentah (station 122 = P.A. Manggarai 2, 6-7 Sep 2026)

```
2026-09-06 00.00.00,491;2026-09-06 00.10.00,496;...;2026-09-07 16.20.00,489;|9500;8500;7500|

<html xmlns="...">
<head><title></title></head>
</html>
```

Struktur: `<titik data dipisah ";">|<siaga1>;<siaga2>;<siaga3>|` diikuti markup
HTML kosong yang harus dibuang saat parsing. Titik data: `yyyy-MM-dd HH.mm.ss,nilai`
(nilai dalam cm papan duga lokal). Ambang siaga datang gratis di respons yang
sama — tidak perlu request terpisah untuk mengisi `stations.yaml`.

Content-Type dari respons: `text/html` (bukan JSON), jadi parser menangani
teks mentah, bukan `json.loads`.

## 2. Nama field form

Tidak relevan lagi karena endpoint GET dipakai langsung. Field asli untuk
referensi (kalau suatu saat endpoint berubah dan butuh fallback VIEWSTATE):
`ctl00$ContentPlaceHolder1$DeWaktuAwal`, `...DeWaktuAkhir`,
`...CmbPintuAir` (DevExpress ASPxComboBox, item value tersembunyi dari DOM,
harus diambil dari page source — lihat §3).

## 3. Daftar pintu air

Kontrol dropdown adalah DevExpress `ASPxComboBox`; item (value+text) di-embed
sebagai literal JS di HTML halaman utama (bukan lazy-load terpisah), pola:
`{'value':'122','text':'P.A. Manggarai 2'}`. Diekstrak 39 pintu air —
tersimpan di [`config/stations.yaml`](../config/stations.yaml). Koordinat
tidak tersedia di portal ini (kosong).

## 4. Batas rentang query

| Rentang | Hasil |
|---|---|
| 1 hari | OK |
| 30 hari | OK, 4467 baris (~1 obs/10 menit, sedikit gap wajar) |
| 90 hari | OK, 13098 baris, respons ~6.2 detik, tidak ada truncation |

Tidak ditemukan batas/timeout sampai 90 hari. **Chunk size dipilih 90 hari**
(satu kuartal) sebagai margin aman — sudah teruji, tanpa perlu mendekati
kemungkinan limit yang belum ditemukan.

## 5. Kedalaman arsip historis

Dites query 1 hari di awal 2024, 2022, dan 2020 (stasiun 122) — ketiganya
mengembalikan data (bukan kosong). Arsip minimal tersedia sejak **2020**.
Tanggal awal arsip yang sebenarnya tidak diuji lebih jauh (di luar kebutuhan
kriteria selesai); jika model butuh data lebih lama, uji ulang mundur dari 2020.

## 6. Interval & anomali

Interval **tidak seragam antar stasiun** — dites 1 hari per stasiun
(7 Sep 2025):

| stasiun | titik/hari | interval implisit |
|---|---|---|
| 122 P.A. Manggarai 2 | ~144 | ~10 menit |
| 169 P.A. Kaliduri 1 | ~288 | ~5 menit |
| 170 P.A. Ancol Flushing 1 / 107 P.A. Cengkareng Drain | ~577 | ~2.5 menit |
| 179/181/187 (Pompa \*, Bubble) | ~125 | ~11.5 menit (dgn gap) |
| 184 Pompa Lagoa (Bubble) | ~167 | ~8.6 menit |

Juga ditemukan gap alami (bukan interval tetap dilanggar rapi) di sampel 2022
dan 2024 (mis. `00.40` lompat ke `01.00`, melewati `00.50`). QC harus
mengasumsikan interval tidak seragam antar stasiun **dan** antar waktu,
bukan resample buta dengan asumsi 10 menit konstan.

**Temuan tambahan (penting untuk §4 chunking):** kepadatan titik data pada
satu stasiun bisa melonjak jauh melebihi baseline harian di atas pada
periode tertentu — retest manual pada stasiun 169 untuk rentang 14 hari
(7-20 Sep 2025, musim yang cenderung basah) mengembalikan ~16.300 titik
(implisit ~1,2 menit/titik, bukan ~5 menit), dan waktu respons servernya naik
non-linear terhadap panjang rentang (1 hari ≈0.8d, 7 hari ≈6.6d, 14 hari
≈20-25d — bukan skala linear ~2 md/hari yang diharapkan). Dugaan: sensor ini
menaikkan frekuensi sampling saat kondisi basah/berpotensi siaga — justru
periode yang paling penting untuk model banjir, dan paling berisiko timeout.
Chunk 90 hari yang aman untuk stasiun 122 (interval stabil, musim kemarau)
menyebabkan `Read timed out` (timeout 30 detik) untuk stasiun 169 pada rentang
Sep-Des 2025.

**Keputusan:** daripada menebak ukuran chunk per stasiun, pipeline dibuat
*adaptif* (`fetch_texts_adaptive` di `src/pipeline.py`) — coba `chunk_days`
dari `settings.yaml` (90 hari) dulu, kalau timeout, chunk dibelah dua dan
dicoba lagi secara rekursif (lantai 2 hari) sampai berhasil. Ini menghindari
harus menebak "chunk aman" per 39 stasiun di muka, dan tetap hemat request
untuk stasiun/periode yang memang jarang.

## 7. Error handling

`IdPintuAir` tidak valid → **HTTP 500** dengan halaman "Runtime Error" ASP.NET
generik (bukan pesan spesifik). Karena `stations.yaml` hanya berisi ID yang
diverifikasi dari dropdown asli, kasus ini seharusnya tidak muncul di operasi
normal; retry logic tetap menghitungnya sebagai 5xx (retryable, maks 3x) demi
kesederhanaan — biaya 3 retry sia-sia untuk kasus yang seharusnya tidak terjadi
lebih murah daripada menulis pengklasifikasi error khusus.

## 8. robots.txt

`https://poskobanjir.dsdadki.web.id/robots.txt` → 404 (tidak ada). Tidak ada
pembatasan crawling yang dinyatakan. Rate limit tetap diterapkan (§4.1 brief)
atas inisiatif sendiri karena ini infrastruktur peringatan dini publik.

## 9. Keputusan desain turunan dari discovery ini

- **Tidak pakai Selenium/Playwright.** Endpoint data adalah GET biasa,
  requests + BeautifulSoup (untuk bersih-bersih tail HTML) sudah cukup.
- **Tidak ada langkah discovery terpisah untuk ambang siaga.** Threshold ikut
  di setiap respons data; pipeline mengisi `stations.yaml` secara otomatis
  saat pull pertama per stasiun alih-alih melakukan 39 request eksplorasi
  terpisah yang tidak perlu.
