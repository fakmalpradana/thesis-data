"""Parser tests against saved fixtures - no network access."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import parse_response  # noqa: E402

FIXTURE = (Path(__file__).parent / "fixtures" / "sample_response.txt").read_text()


def test_parses_data_points():
    df, thresholds = parse_response(FIXTURE, station_id=122, station_nama="P.A. Manggarai 2")
    assert len(df) == 10
    assert list(df.columns) == ["waktu", "stasiun_id", "stasiun_nama", "tma_cm"]
    assert df["tma_cm"].iloc[0] == 491.0
    assert df["stasiun_id"].iloc[0] == "122"
    assert str(df["waktu"].dt.tz) == "Asia/Jakarta"


def test_parses_thresholds_divided_by_ten():
    _, thresholds = parse_response(FIXTURE, station_id=122, station_nama="P.A. Manggarai 2")
    assert thresholds == {"siaga1": 950.0, "siaga2": 850.0, "siaga3": 750.0}


def test_empty_data_segment_does_not_crash():
    raw = "|9500;8500;7500|\n\n<html></html>"
    df, thresholds = parse_response(raw, station_id=122, station_nama="x")
    assert df.empty
    assert thresholds["siaga1"] == 950.0


if __name__ == "__main__":
    test_parses_data_points()
    test_parses_thresholds_divided_by_ten()
    test_empty_data_segment_does_not_crash()
    print("ok")
