"""One-off: fill tinggi_air/config/stations.yaml `koordinat` for the 14 Jakut
TMA stations by fuzzy-matching their poskobanjir names to DSDA satudata
pintu-air / rumah-pompa coordinate tables. Idempotent (re-run = re-match +
overwrite the same 14 entries).

    python3 scripts/match_station_coords.py [--write]

Without --write: prints the match table only (dry run). With --write: also
updates stations.yaml.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "tinggi_air/config/stations.yaml"

JAKUT = [107, 126, 140, 150, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187]
CUTOFF = 0.6
BBOX = (-6.35, -5.95, 106.6, 107.05)  # lat_min, lat_max, lon_min, lon_max

# 140/166: two gauges (laut/kali) at the same physical pintu air -> same coords.
MANUAL = {166: 140}

STRIP_WORDS = ["p.a.", "p.s.", "pompa", "rumah pompa", "pintu air", "bendung.", "waduk"]
STRIP_PAREN = ["(bubble)", "(laut)", "(kali)"]


def normalize(name: str) -> str:
    s = name.lower()
    for p in STRIP_PAREN:
        s = s.replace(p, "")
    for w in STRIP_WORDS:
        s = s.replace(w, "")
    s = re.sub(r"\d+", "", s)          # trailing gauge numbers (Pulo Gadung 2 -> ...)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def best_match(query: str, choices: dict[str, tuple]) -> tuple[str | None, float, tuple | None]:
    """choices: normalized_name -> (orig_name, lat, lon). Returns (orig_name, score, (lat,lon))."""
    names = list(choices)
    hits = difflib.get_close_matches(query, names, n=1, cutoff=CUTOFF)
    if not hits:
        return None, 0.0, None
    hit = hits[0]
    score = difflib.SequenceMatcher(None, query, hit).ratio()
    orig, lat, lon = choices[hit]
    return orig, score, (lat, lon)


def load_choices(df: pd.DataFrame, name_col: str) -> dict[str, tuple]:
    out = {}
    for _, r in df.iterrows():
        try:
            lat, lon = float(r["lintang"]), float(r["bujur"])
        except (TypeError, ValueError):
            continue
        norm = normalize(str(r[name_col]))
        if norm and norm not in out:  # first occurrence wins (dupes across periode_data)
            out[norm] = (r[name_col], lat, lon)
    return out


def in_bbox(lat: float, lon: float) -> bool:
    lat_min, lat_max, lon_min, lon_max = BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def main() -> None:
    write = "--write" in sys.argv
    pintu_air = pd.read_parquet(ROOT / "validation/data/processed/data-pintu-air.parquet")
    rumah_pompa = pd.read_parquet(ROOT / "validation/data/processed/data-lokasi-rumah-pompa.parquet")
    pintu_choices = load_choices(pintu_air, "nama_pintu_air")
    pompa_choices = load_choices(rumah_pompa, "lokasi")

    doc = yaml.safe_load(YAML_PATH.read_text())
    stations = {x["id"]: x for x in doc["stations"]}

    results = {}
    rows = []
    for sid in JAKUT:
        st = stations[sid]
        if sid in MANUAL:
            src_id = MANUAL[sid]
            results[sid] = results[src_id]
            orig, score, source = results[src_id]["match"], results[src_id]["score"], results[src_id]["source"]
            lat, lon = results[src_id]["lat"], results[src_id]["lon"]
            rows.append((sid, st["nama"], orig, score, lat, lon, source + " (shared w/ " + str(src_id) + ")"))
            continue
        q = normalize(st["nama"])
        cand_pa = (*best_match(q, pintu_choices), "dsda_pintu_air")
        cand_rp = (*best_match(q, pompa_choices), "dsda_rumah_pompa")
        # best score wins, not first-source-wins - a high-scoring pintu_air
        # false-friend (e.g. "Kali Asin" -> "Kali Mati") must lose to an
        # exact rumah_pompa hit.
        orig, score, latlon, source = max((cand_pa, cand_rp), key=lambda c: c[1])
        if orig is None or score < CUTOFF or not in_bbox(*latlon):
            results[sid] = {"match": None, "score": 0.0, "lat": None, "lon": None, "source": None}
            rows.append((sid, st["nama"], None, 0.0, None, None, "NO MATCH"))
            continue
        lat, lon = latlon
        results[sid] = {"match": orig, "score": score, "lat": lat, "lon": lon, "source": source}
        rows.append((sid, st["nama"], orig, score, lat, lon, source))

    hdr = f"{'id':>4} {'station':<32} {'matched':<28} {'score':>5} {'lat':>10} {'lon':>11}  source"
    print(hdr)
    print("-" * len(hdr))
    n_ok = 0
    for sid, nama, orig, score, lat, lon, source in rows:
        if lat is not None:
            n_ok += 1
        print(f"{sid:>4} {nama:<32} {str(orig):<28} {score:>5.2f} "
              f"{lat if lat is not None else '':>10} {lon if lon is not None else '':>11}  {source}")
    print(f"\n{n_ok}/{len(JAKUT)} matched (score >= {CUTOFF}, inside Jakarta bbox)")

    if not write:
        print("\n(dry run - pass --write to update stations.yaml)")
        return

    for sid, r in results.items():
        if r["lat"] is None:
            continue
        sumber = f"DSDA satudata {r['source']} (by-name, score {r['score']:.2f})"
        stations[sid]["koordinat"] = {"lat": round(r["lat"], 6), "lon": round(r["lon"], 6), "sumber": sumber}

    YAML_PATH.write_text(yaml.dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False))
    print(f"\nwrote {YAML_PATH}")


if __name__ == "__main__":
    main()
