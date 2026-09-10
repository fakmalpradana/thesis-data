"""Fetch BPBD/DSDA open datasets from satudata.jakarta.go.id (undocumented
backend used by the SPA; pagination = data_no/per_page, max 100/page).

    python3 validation/src/fetch_satudata.py          # all datasets in DATASETS
    python3 validation/src/fetch_satudata.py <slug>   # one

Raw JSON -> data/raw/<slug>.json ; flat table -> data/processed/<slug>.parquet
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://satudata.jakarta.go.id/backend/api/v2/satudata/detail"
HDR = {"content-type": "application/json", "origin": "https://satudata.jakarta.go.id",
       "referer": "https://satudata.jakarta.go.id/", "user-agent": "Mozilla/5.0 (thesis-data fetch)"}
ROOT = Path(__file__).resolve().parent.parent

DATASETS = [  # slug: BPBD flood-event tables per year + DSDA infrastructure with coordinates
    "data-kejadian-bencana-banjir-di-provinsi-dki-jakarta-tahun-2019",
    "data-kejadian-bencana-banjir-di-provinsi-dki-jakarta-tahun-2020",
    "data-kejadian-bencana-di-provinsi-dki-jakarta-tahun-2021",
    "data-kejadian-bencana-banjir-tahun-2023",
    "data-kejadian-bencana-banjir-tahun-2024",
    "data-kejadian-bencana-banjir",                       # 2025Q1 - 2026Q1 (rolling)
    "data-rekapitulasi-tahunan-kejadian-banjir-di-provinsi-dki-jakarta",  # 2013-2020 annual
    "luasan-daerah-tergenang-tahun-2023",
    "data-pintu-air",
    "data-lokasi-rumah-pompa",
    "data-titik-rawan-bencanabanjir",
]


def fetch(slug: str) -> tuple[dict, list[dict]]:
    meta, rows = None, []
    for page in range(1, 200):
        for attempt in range(5):  # ponytail: server drops TLS handshakes; plain retry, no backoff lib
            try:
                r = requests.post(API, headers=HDR, timeout=90,
                                  json={"page_url": slug, "kategori": "dataset", "data_no": page, "per_page": 100})
                r.raise_for_status()
                j = r.json()
                break
            except requests.RequestException as e:
                print(f"  retry {attempt + 1} {slug} p{page}: {type(e).__name__}")
                time.sleep(5)
        else:
            raise RuntimeError(f"gave up on {slug} page {page}")
        meta = meta or {k: j["data"].get(k) for k in ("title", "desc", "sumber", "created_at", "updated_at", "kontak")}
        f = j.get("filedata") or []
        rows += f
        if len(f) < 100:
            break
        time.sleep(0.3)
    return meta, rows


def main(slugs: list[str]) -> None:
    for slug in slugs:
        if (ROOT / "data/processed" / f"{slug}.parquet").exists() and "--force" not in sys.argv:
            print(f"{slug}: exists, skip")
            continue
        meta, rows = fetch(slug)
        (ROOT / "data/raw" / f"{slug}.json").write_text(
            json.dumps({"meta": meta, "fetched_at": time.strftime("%Y-%m-%d"), "rows": rows}, ensure_ascii=False))
        df = pd.DataFrame(rows).drop(columns=["rn"], errors="ignore")
        df.to_parquet(ROOT / "data/processed" / f"{slug}.parquet", index=False)
        print(f"{slug}: {len(df)} rows, cols={list(df.columns)[:6]}...")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or DATASETS)
