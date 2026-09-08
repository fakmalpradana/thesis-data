"""Tide prediction tests - runs against the real local EOT20 data (not a
network hit; the 17x130MB constituent set isn't fixture-able, see tide.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # noqa: E402
from sources import tide  # noqa: E402


def test_nearest_wet_point_avoids_land_masked_coordinate():
    # Marina Ancol's own coordinate is land-masked at EOT20's 1/8deg grid.
    lon, lat = tide.nearest_wet_point(106.8291, -6.1215)
    assert (lon, lat) == (106.875, -6.0)


def test_predict_returns_no_nan_at_wet_point():
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="Asia/Jakarta")
    s = tide.predict(106.875, -6.0, idx)
    assert s.notna().all()
    assert s.abs().max() < 2.0


if __name__ == "__main__":
    test_nearest_wet_point_avoids_land_masked_coordinate()
    test_predict_returns_no_nan_at_wet_point()
    print("ok")
