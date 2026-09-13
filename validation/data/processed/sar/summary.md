# SAR flood extent — validation summary
Fix: land clip (Kota Adm. Jakarta Utara), sea/permanent-water mask (dry-ref water OR DTM<=0.3m),
5x5 despeckle, min-polygon 0.25ha, Otsu on land pixels only, orbit-agnostic scene pick (smallest
gap within +/-4d).

Effect: flood area dropped ~100x vs. pre-fix (was 1,235/1,251 ha, dominated by Teluk Jakarta sea)
to 0.29 km2 (E2), 0.12 km2 (E6), 0.12 km2 (E7) — plausible urban-flood magnitudes.

Otsu-on-land: -3.7 to -4.3 dB for all three events (urban backscatter still dominates), out of
the [-22,-13] band, so the -16 dB fallback was used throughout, as designed.

E2 orbit: chosen ascending scene at 23.8h gap; only alternative within +/-4d was descending at
60.9h — ascending is the correct (smaller-gap) pick.

PetaBencana hit rate (100m buffer, +/-1d window): E2 0/9 (0.0%), E6 0/32 (0.0%), E7 0/1 (0.0%).
Per-kecamatan flood_ha vs bpbd_kejadian: flooded BPBD kecamatan (Cilincing, Penjaringan) do get
nonzero SAR flood_ha, but polygons miss individual PetaBencana points.

**Verdict**: 0% of reports within 100m for every event, well under the 30% usability bar.
SAR C-band tidak menangkap genangan dangkal Jakut untuk event ini.
