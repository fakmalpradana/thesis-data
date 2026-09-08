"""QC flag tests - synthetic data, no network access."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qc import apply_qc, resample_hourly  # noqa: E402


def _mk(values, freq="10min"):
    idx = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="Asia/Jakarta")
    return pd.DataFrame(
        {"waktu": idx, "stasiun_id": "1", "stasiun_nama": "test", "tma_cm": values}
    )


def test_sentinel_flagged_1_but_plain_negative_is_not():
    # -999 is the known sentinel; a plain negative reading is NOT flagged by
    # itself - per-station papan duga datums are arbitrary (brief §7), some
    # stations read consistently negative under normal conditions.
    df = apply_qc(_mk([100, -999, 105, -5, 102]))
    assert df.loc[df["tma_cm"] == -999, "qc_flag"].iloc[0] == 1
    assert df.loc[df["tma_cm"] == -5, "qc_flag"].iloc[0] == 0


def test_stuck_sensor_flagged_2():
    df = apply_qc(_mk([100] * 8 + [101, 102]))
    assert (df["qc_flag"].iloc[:8] == 2).all()
    assert df["qc_flag"].iloc[8] == 0


def test_duplicate_timestamp_flagged_4():
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 00:10"]).tz_localize(
        "Asia/Jakarta"
    )
    df = pd.DataFrame({"waktu": idx, "stasiun_id": "1", "stasiun_nama": "t", "tma_cm": [100, 100, 101]})
    out = apply_qc(df)
    assert out["qc_flag"].tolist() == [0, 4, 0]


def test_hourly_resample_flags_gap():
    df = apply_qc(_mk([100, 101], freq="70min"))  # only 1 obs per hour -> gap
    hourly = resample_hourly(df)
    assert (hourly["qc_flag"] == 9).any()


def test_hourly_resample_flags_degraded_when_mostly_stuck():
    # Bypass apply_qc (a >60min stuck run can't fit inside a single 60min
    # hour bucket in isolation - tested for real via apply_qc separately
    # above) and test resample_hourly's propagation logic directly: 8 of 10
    # points in the hour already point-flagged -> pct_flagged=0.8 -> qc_flag=8.
    df = _mk([100] * 10, freq="6min")
    df["qc_flag"] = [2] * 8 + [0, 0]
    hourly = resample_hourly(df)
    assert len(hourly) == 1
    assert hourly["qc_flag"].iloc[0] == 8
    assert hourly["pct_flagged"].iloc[0] == 0.8
    # tma_mean/tma_max still computed from all 10 points - flagged, not dropped.
    assert hourly["n_obs"].iloc[0] == 10


def test_hourly_resample_ok_when_mostly_clean():
    df = apply_qc(_mk([100, 101, 102, 103, 104], freq="10min"))
    hourly = resample_hourly(df)
    assert hourly["qc_flag"].iloc[0] == 0
    assert hourly["pct_flagged"].iloc[0] == 0.0


if __name__ == "__main__":
    test_sentinel_flagged_1_but_plain_negative_is_not()
    test_stuck_sensor_flagged_2()
    test_duplicate_timestamp_flagged_4()
    test_hourly_resample_flags_gap()
    test_hourly_resample_flags_degraded_when_mostly_stuck()
    test_hourly_resample_ok_when_mostly_clean()
    print("ok")
