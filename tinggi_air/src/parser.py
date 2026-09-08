"""Raw response -> DataFrame.

Raw format (docs/DISCOVERY.md §1):
    "yyyy-MM-dd HH.mm.ss,val;yyyy-MM-dd HH.mm.ss,val;...;|siaga1;siaga2;siaga3|\n\n<html>...garbage"
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


class ParseError(Exception):
    pass


def parse_response(raw: str, station_id: int, station_nama: str) -> tuple[pd.DataFrame, dict]:
    """Returns (dataframe, thresholds). thresholds = {siaga1, siaga2, siaga3} in cm, or
    all-None if the response carried none (e.g. genuinely empty range)."""
    if "|" not in raw:
        raise ParseError(f"unexpected response shape for station {station_id}: {raw[:200]!r}")

    data_part, threshold_part, _tail = raw.split("|", 2) if raw.count("|") >= 2 else (*raw.split("|", 1), "")

    rows = []
    for point in data_part.split(";"):
        point = point.strip()
        if not point:
            continue
        ts_str, _, val_str = point.partition(",")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H.%M.%S")
            val = float(val_str)
        except ValueError as exc:
            raise ParseError(f"bad data point {point!r} for station {station_id}: {exc}") from exc
        rows.append((ts, val))

    df = pd.DataFrame(rows, columns=["waktu", "tma_cm"])
    if not df.empty:
        df["waktu"] = pd.to_datetime(df["waktu"]).dt.tz_localize("Asia/Jakarta")
    df["stasiun_id"] = str(station_id)
    df["stasiun_nama"] = station_nama
    df = df[["waktu", "stasiun_id", "stasiun_nama", "tma_cm"]]

    thresholds = {"siaga1": None, "siaga2": None, "siaga3": None}
    parts = [p for p in threshold_part.split(";") if p.strip()]
    if len(parts) == 3:
        try:
            thresholds = {
                "siaga1": float(parts[0]) / 10.0,
                "siaga2": float(parts[1]) / 10.0,
                "siaga3": float(parts[2]) / 10.0,
            }
        except ValueError:
            pass  # leave as None - don't guess

    return df, thresholds
