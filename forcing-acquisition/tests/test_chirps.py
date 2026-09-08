"""CHIRPS parse() test - local fixture, no network access."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sources import chirps  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "chirps_sample.nc"


def test_parse_returns_expected_shape_and_units():
    ds = chirps.parse(FIXTURE)
    assert ds["precip"].attrs.get("units") == "mm/day"
    assert ds.sizes["time"] == 10
    assert float(ds.latitude.values[0]) < float(ds.latitude.values[-1])  # south-up


if __name__ == "__main__":
    test_parse_returns_expected_shape_and_units()
    print("ok")
