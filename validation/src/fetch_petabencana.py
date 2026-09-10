"""Fetch PetaBencana.id crowdsourced reports (archive API, monthly chunks) for
DKI Jakarta, 2021-01 -> today. Point reports with disaster_type, flood depth
(report_data.flood_depth, cm), timestamp, kelurahan code.

    python3 validation/src/fetch_petabencana.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://data.petabencana.id/reports/archive"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/petabencana_reports_jakarta.parquet"


def main() -> None:
    months = pd.date_range("2021-01-01", pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1), freq="MS")
    feats = []
    for a, b in zip(months[:-1], months[1:]):
        for attempt in range(4):
            try:
                r = requests.get(API, timeout=120, params={
                    "start": a.strftime("%Y-%m-%dT00:00:00+0700"), "end": b.strftime("%Y-%m-%dT00:00:00+0700"),
                    "geoformat": "geojson", "admin": "ID-JK"})
                r.raise_for_status()
                f = (r.json().get("result") or {}).get("features") or []
                break
            except requests.RequestException as e:
                print(f"  retry {attempt + 1} {a:%Y-%m}: {type(e).__name__}"); time.sleep(5)
        else:
            raise RuntimeError(f"gave up {a:%Y-%m}")
        feats += f
        print(f"{a:%Y-%m}: {len(f)}")
        time.sleep(0.5)
    (ROOT / "data/raw/petabencana_reports_jakarta.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}))
    rows = []
    for x in feats:
        p, rd = x["properties"], (x["properties"].get("report_data") or {})
        rows.append({"pkey": p.get("pkey"), "created_at": p.get("created_at"), "disaster_type": p.get("disaster_type"),
                     "status": p.get("status"), "source": p.get("source"), "flood_depth_cm": rd.get("flood_depth"),
                     "text": p.get("text"), "city": (p.get("tags") or {}).get("city"),
                     "local_area_id": (p.get("tags") or {}).get("local_area_id"),
                     "lon": x["geometry"]["coordinates"][0], "lat": x["geometry"]["coordinates"][1]})
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df.to_parquet(OUT, index=False)
    print(f"\n{len(df)} reports -> {OUT}")
    print(df.disaster_type.value_counts().to_string())


if __name__ == "__main__":
    main()
