"""Gabungkan TMA (target) + hujan (CHIRPS) + pasut (EOT20) jadi satu tabel
jam-an ter-align untuk satu node MVP (brief §5). Tidak ada gap tersembunyi:
jam tanpa data tetap jadi baris NaN, bukan hilang dari index.

Konvensi waktu (brief §5, wajib disamakan - inilah bagian paling gampang salah):
  - Index akhir: jam-an, Asia/Jakarta, label = awal interval (jam 00:00
    mewakili 00:00-01:00), sama seperti resample TMA di ../tinggi_air/src/qc.py.
  - CHIRPS: satu nilai per hari, TIDAK diinterpolasi jadi jam-an (dilarang
    §5) - di-forward-fill ke 24 jam yang termasuk dalam hari itu, dengan
    kolom `hujan_resolusi='harian'` sebagai flag eksplisit.
  - **Asumsi yang harus dinyatakan di tesis**: "hari" CHIRPS diasumsikan
    hari kalender UTC (00:00-24:00 UTC), konvensi umum produk presipitasi
    satelit global - TIDAK diverifikasi eksplisit ke dokumentasi CHIRPS
    (SOURCE_NOTES.md tidak menemukan pernyataan timezone eksplisit, hanya
    "first day of the given day"). Kalau asumsi ini salah, forcing hujan
    bisa bergeser hingga 7 jam (WIB=UTC+7) terhadap TMA - cukup untuk
    merusak lag structure (persis peringatan brief §5). Pemetaan
    UTC-day -> jam WIB dilakukan eksplisit di bawah (bukan asumsi hari
    kalender WIB) supaya pergeseran ini sudah diperhitungkan, bukan
    diabaikan.
  - Pasut (EOT20): sudah dihitung langsung di titik jam bulat WIB, tidak
    perlu resample.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

import qc

ROOT = Path(__file__).resolve().parent.parent
TMA_ROOT = ROOT.parent / "tinggi_air"


def load_tma_hourly(station_id: int) -> pd.DataFrame:
    files = sorted(TMA_ROOT.glob(f"data/processed/{station_id}_*_hourly.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values("waktu")
    df["waktu"] = pd.to_datetime(df["waktu"])
    return df[["waktu", "tma_mean", "tma_max", "n_obs", "pct_flagged", "qc_flag"]].rename(
        columns={"n_obs": "tma_n_obs", "pct_flagged": "tma_pct_flagged", "qc_flag": "tma_qc_flag"}
    )


def load_chirps_daily_utc(nc_path: Path) -> pd.Series:
    """Basin-averaged (bbox mean, see domain.yaml caveat - not a true DAS
    mask) daily rainfall, indexed by UTC day-start timestamp."""
    ds = xr.open_dataset(nc_path)
    daily = ds["precip"].mean(dim=["latitude", "longitude"])
    s = daily.to_series()
    s.index = pd.to_datetime(s.index).tz_localize("UTC")  # see module docstring assumption
    s.name = "hujan_mm"
    return s


def rain_to_hourly(daily_utc: pd.Series, hourly_index_wib: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill each UTC calendar day's value onto the WIB hours whose
    UTC instant falls inside that day - not onto the WIB calendar day (see
    module docstring). Never interpolated, always the same flat value across
    the (up to 24) hours it actually covers - flagged explicitly."""
    hourly_utc = hourly_index_wib.tz_convert("UTC")
    day_starts = daily_utc.index
    pos = day_starts.searchsorted(hourly_utc, side="right") - 1
    valid = (pos >= 0) & (pos < len(day_starts))
    values = pd.Series(index=hourly_index_wib, dtype="float64")
    values.iloc[valid] = daily_utc.values[pos[valid]]
    return pd.DataFrame(
        {
            "hujan_mm": values,
            "hujan_resolusi": "harian",
            "hujan_qc_flag": qc.flag_rain(values),
        },
        index=hourly_index_wib,
    )


def build(station_id: int, chirps_nc: Path, tide_parquet: Path) -> pd.DataFrame:
    tma = load_tma_hourly(station_id).set_index("waktu")

    full_index = pd.date_range(tma.index.min(), tma.index.max(), freq="h", tz="Asia/Jakarta")
    tma = tma.reindex(full_index)  # brief §5: no hidden gaps, absent hours become NaN rows

    rain = rain_to_hourly(load_chirps_daily_utc(chirps_nc), full_index)

    tide = pd.read_parquet(tide_parquet).set_index("waktu")
    tide.index = pd.to_datetime(tide.index)
    tide = tide.reindex(full_index)
    tide["tide_qc_flag"] = qc.flag_tide(tide["tide_m"])

    out = tma.join(rain).join(tide)
    out.index.name = "waktu"
    return out.reset_index()


def build_report(df: pd.DataFrame) -> str:
    """Brief §7: missing rate per bulan per variabel + gap terpanjang."""
    d = df.copy()
    d["bulan"] = d["waktu"].dt.tz_localize(None).dt.to_period("M")
    cols = ["tma_mean", "hujan_mm", "tide_m"]
    lines = [f"# Laporan Forcing - {pd.Timestamp.today().date().isoformat()}", ""]
    lines.append("## Missing rate per bulan per variabel")
    lines.append("")
    lines.append("| bulan | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for bulan, g in d.groupby("bulan"):
        rates = [f"{g[c].isna().mean():.1%}" for c in cols]
        lines.append(f"| {bulan} | " + " | ".join(rates) + " |")

    lines += ["", "## qc_flag distribution", ""]
    lines.append("tma_qc_flag: " + d["tma_qc_flag"].value_counts(dropna=False).sort_index().to_dict().__repr__())
    lines.append("hujan_qc_flag: " + d["hujan_qc_flag"].value_counts(dropna=False).sort_index().to_dict().__repr__())
    lines.append("tide_qc_flag: " + d["tide_qc_flag"].value_counts(dropna=False).sort_index().to_dict().__repr__())

    lines += ["", "## Gap terpanjang (tma_mean kosong)", ""]
    is_gap = d["tma_mean"].isna()
    gap_id = (is_gap != is_gap.shift()).cumsum()
    gaps = d[is_gap].groupby(gap_id)["waktu"].agg(["min", "max", "count"]).sort_values("count", ascending=False).head(5)
    for _, row in gaps.iterrows():
        lines.append(f"- {row['min']} .. {row['max']} ({int(row['count'])} jam)")

    return "\n".join(lines)


if __name__ == "__main__":
    with open(ROOT / "config" / "nodes.yaml") as f:
        node = yaml.safe_load(f)["mvp_node"]

    chirps_files = sorted((ROOT / "data" / "raw" / "chirps").glob("chirps_*.nc"))
    chirps_nc = max(chirps_files, key=lambda p: p.stat().st_size)  # the full-period pull, not the 10-day smoke test
    tide_parquet = ROOT / "data" / "processed" / f"tide_{node['tma_stasiun_id']}_2021_2026.parquet"

    df = build(node["tma_stasiun_id"], chirps_nc, tide_parquet)
    out_path = ROOT / "data" / "processed" / "forcing_hourly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved {out_path}, {len(df)} rows, {df['waktu'].min()} .. {df['waktu'].max()}")
    print(df.isna().mean().round(3))

    report = build_report(df)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"forcing_report_{pd.Timestamp.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report)
    print(f"report: {report_path}")
