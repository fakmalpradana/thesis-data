"""OSM drainage parse() test - local fixture, no network access."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sources import osm_drainage  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "osm_drainage_sample.json"


def test_parse_returns_elements_list():
    els = osm_drainage.parse(FIXTURE)
    assert len(els) > 0
    assert all("type" in e for e in els)


if __name__ == "__main__":
    test_parse_returns_elements_list()
    print("ok")
