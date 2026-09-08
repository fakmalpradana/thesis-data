"""Orchestration: chunked, resumable, rate-limited acquisition + QC + report.

Idempotency/resume (brief §4.1, kriteria selesai #3): a fetched chunk's raw
response is saved to data/raw/{station}/{start}_{end}.txt *before* parsing.
That file's existence on disk *is* the "already fetched" marker - no separate
manifest needed. Re-running after a crash just skips chunks whose raw file
is already there and re-reads it from disk instead of hitting the network.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from client import FetchError, Settings, TmaClient  # noqa: E402
from parser import parse_response  # noqa: E402
from qc import apply_qc, resample_hourly  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def load_settings() -> Settings:
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    return Settings(
        base_url=cfg["base_url"],
        user_agent=cfg["user_agent"],
        chunk_days=cfg["chunk_days"],
        delay_min_s=cfg["delay_min_s"],
        delay_max_s=cfg["delay_max_s"],
        max_retries=cfg["max_retries"],
        retry_backoff_base_s=cfg["retry_backoff_base_s"],
        request_timeout_s=cfg["request_timeout_s"],
    ), cfg


def load_stations() -> dict:
    with open(ROOT / "config" / "stations.yaml") as f:
        return yaml.safe_load(f)


def save_stations(stations_cfg: dict) -> None:
    with open(ROOT / "config" / "stations.yaml", "w") as f:
        yaml.safe_dump(stations_cfg, f, allow_unicode=True, sort_keys=False)


def chunk_ranges(start: date, end: date, chunk_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def raw_path(station_id: int, start: date, end: date) -> Path:
    d = RAW_DIR / str(station_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{start.isoformat()}_{end.isoformat()}.txt"


MIN_SPLIT_DAYS = 2  # floor before giving up and letting the error propagate


def fetch_texts_adaptive(client: TmaClient, station_id: int, start: date, end: date) -> list[str]:
    """Fetch one chunk; on a timeout, split it in half and recurse instead of
    guessing a chunk size upfront. Some stations log far more densely than
    others (2.5min vs 5min vs 10min - see DISCOVERY.md §6) and the server's
    response time grows worse than linearly with range, so a size that's fine
    for one station can time out for another. Self-tunes per station/period
    instead."""
    path = raw_path(station_id, start, end)
    if path.exists():
        return [path.read_text()]
    try:
        text = client.fetch_raw(station_id, start, end)
    except FetchError as exc:
        span_days = (end - start).days + 1
        if span_days > MIN_SPLIT_DAYS and "timed out" in str(exc).lower():
            mid = start + timedelta(days=span_days // 2 - 1)
            print(f"  [{station_id}] {start}..{end} timed out, splitting at {mid}")
            return fetch_texts_adaptive(client, station_id, start, mid) + fetch_texts_adaptive(
                client, station_id, mid + timedelta(days=1), end
            )
        raise
    path.write_text(text)
    return [text]


def last_saved_timestamp(station_id: int) -> pd.Timestamp | None:
    files = sorted(PROCESSED_DIR.glob(f"{station_id}_*.parquet"))
    files = [f for f in files if "_hourly" not in f.name]
    latest = None
    for f in files:
        df = pd.read_parquet(f, columns=["waktu"])
        if df.empty:
            continue
        m = df["waktu"].max()
        if latest is None or m > latest:
            latest = m
    return latest


def process_station(client: TmaClient, station: dict, start: date, end: date, stations_cfg: dict) -> dict:
    station_id, nama = station["id"], station["nama"]
    chunks_raw = []
    for c_start, c_end in chunk_ranges(start, end, client.settings.chunk_days):
        chunks_raw.extend(fetch_texts_adaptive(client, station_id, c_start, c_end))

    frames, thresholds = [], None
    for text in chunks_raw:
        df, th = parse_response(text, station_id, nama)
        if not df.empty:
            frames.append(df)
        if thresholds is None and any(v is not None for v in th.values()):
            thresholds = th

    if thresholds:
        for s in stations_cfg["stations"]:
            if s["id"] == station_id and any(v is None for v in s["ambang_siaga_cm"].values()):
                s["ambang_siaga_cm"] = thresholds

    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["waktu", "stasiun_id", "stasiun_nama", "tma_cm"]
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    written_years = []
    for year, g in new_df.groupby(new_df["waktu"].dt.year) if not new_df.empty else []:
        path = PROCESSED_DIR / f"{station_id}_{year}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing.drop(columns=["qc_flag"], errors="ignore"), g], ignore_index=True)
        else:
            merged = g
        merged = merged.drop_duplicates(subset=["waktu"]).sort_values("waktu").reset_index(drop=True)
        merged = apply_qc(merged)
        merged.to_parquet(path, index=False)

        hourly = resample_hourly(merged)
        hourly.to_parquet(PROCESSED_DIR / f"{station_id}_{year}_hourly.parquet", index=False)
        written_years.append(int(year))

    return {"station_id": station_id, "nama": nama, "n_new_obs": len(new_df), "years": written_years}


def parse_ddmmyyyy(s: str) -> date:
    return datetime.strptime(s, "%d/%m/%Y").date()


def build_report(run_summaries: list[dict], start: date, end: date) -> str:
    lines = [
        f"# Laporan Akuisisi TMA - {date.today().isoformat()}",
        "",
        f"Periode ditarik: {start.isoformat()} s/d {end.isoformat()}",
        "",
        "## Ringkasan per stasiun",
        "",
        "| stasiun_id | nama | observasi baru | tahun ditulis |",
        "|---|---|---|---|",
    ]
    for s in run_summaries:
        lines.append(f"| {s['station_id']} | {s['nama']} | {s['n_new_obs']} | {s['years']} |")

    lines += ["", "## QC & missing rate per bulan", ""]
    for s in run_summaries:
        for year in s["years"]:
            path = PROCESSED_DIR / f"{s['station_id']}_{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df["bulan"] = df["waktu"].dt.tz_localize(None).dt.to_period("M")
            lines.append(f"### {s['nama']} ({s['station_id']}) - {year}")
            lines.append("")
            lines.append("qc_flag distribution: " + df["qc_flag"].value_counts().sort_index().to_dict().__repr__())
            lines.append("")
            lines.append("| bulan | n_obs | expected (~10min) | missing rate |")
            lines.append("|---|---|---|---|")
            for bulan, g in df.groupby("bulan"):
                days_in_month = bulan.days_in_month
                expected = days_in_month * 24 * 6
                n_obs = len(g)
                missing_rate = max(0.0, 1 - n_obs / expected)
                lines.append(f"| {bulan} | {n_obs} | {expected} | {missing_rate:.1%} |")
            lines.append("")

            gaps = df["waktu"].sort_values().diff().dropna()
            if not gaps.empty:
                top_gaps = gaps.sort_values(ascending=False).head(5)
                lines.append("Gap terpanjang: " + ", ".join(str(g) for g in top_gaps))
                lines.append("")

    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Akuisisi TMA pintu air DKI Jakarta")
    ap.add_argument("--stations", help="comma-separated station id, default: semua di stations.yaml")
    ap.add_argument("--start", help="dd/mm/yyyy, default dari settings.yaml")
    ap.add_argument("--end", help="dd/mm/yyyy, default dari settings.yaml")
    ap.add_argument("--incremental", action="store_true", help="tarik hanya sejak timestamp terakhir tersimpan")
    ap.add_argument(
        "--chunk-days",
        type=int,
        help="override chunk_days dari settings.yaml (stasiun berinterval rapat/5-menit "
        "butuh chunk lebih kecil dari 90 hari agar tidak timeout - lihat DISCOVERY.md",
    )
    ap.add_argument("--timeout-s", type=float, help="override request_timeout_s dari settings.yaml")
    args = ap.parse_args(argv)

    settings, cfg = load_settings()
    if args.chunk_days:
        settings.chunk_days = args.chunk_days
    if args.timeout_s:
        settings.request_timeout_s = args.timeout_s
    stations_cfg = load_stations()
    client = TmaClient(settings)

    all_stations = stations_cfg["stations"]
    if args.stations:
        wanted = {int(x) for x in args.stations.split(",")}
        all_stations = [s for s in all_stations if s["id"] in wanted]

    default_end = parse_ddmmyyyy(cfg["default_end"])
    default_start = parse_ddmmyyyy(cfg["default_start"])
    end = parse_ddmmyyyy(args.end) if args.end else default_end
    report_start = parse_ddmmyyyy(args.start) if args.start else default_start

    summaries = []
    for station in all_stations:
        if args.incremental:
            last = last_saved_timestamp(station["id"])
            start = (last.date() if last is not None else default_start)
        else:
            start = parse_ddmmyyyy(args.start) if args.start else default_start
        if start > end:
            continue
        print(f"[{station['id']}] {station['nama']}: {start} .. {end}")
        summaries.append(process_station(client, station, start, end, stations_cfg))
        save_stations(stations_cfg)  # persist after each station, not just at the end -
        # a crash mid-run should not lose thresholds already discovered for earlier stations

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(summaries, report_start, end)
    report_path = REPORTS_DIR / f"run_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
