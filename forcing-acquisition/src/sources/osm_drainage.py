"""OSM drainage network (kanal, sungai, tanggul) via Overpass API. No credentials.
Verified interactively (docs/SOURCE_NOTES.md §3): plain POST with an Overpass
QL query, JSON response, no auth. One-time static pull, not scheduled.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from .base import BBox

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "osm_drainage"
USER_AGENT = "ThesisResearchBot/0.1 (+contact: fakmalpradana@gmail.com; academic use, UGM thesis)"


def fetch(bbox: BBox) -> Path:
    """Static pull - waterway=* (river/canal/drain/ditch) + man_made=embankment
    (tanggul) inside bbox. Saved as-is (raw Overpass JSON)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"osm_drainage_{bbox.lon_min}_{bbox.lat_min}_{bbox.lon_max}_{bbox.lat_max}.json"
    if out_path.exists():
        return out_path

    query = (
        "[out:json][timeout:60];"
        f"(way[\"waterway\"]({bbox.lat_min},{bbox.lon_min},{bbox.lat_max},{bbox.lon_max});"
        f"way[\"man_made\"=\"embankment\"]({bbox.lat_min},{bbox.lon_min},{bbox.lat_max},{bbox.lon_max});"
        ");out geom;"
    )
    resp = requests.post(
        OVERPASS_URL, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    resp.raise_for_status()
    out_path.write_text(resp.text)
    return out_path


def parse(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["elements"]
