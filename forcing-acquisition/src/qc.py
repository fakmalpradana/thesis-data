"""QC: flag, never silently drop/fill (brief §7), mirrors ../tinggi_air/src/qc.py philosophy."""
from __future__ import annotations

import pandas as pd

RAIN_MAX_PLAUSIBLE_MM_DAY = 300.0  # generous global daily-record margin; calibrate down if data shows otherwise
TIDE_MAX_PLAUSIBLE_M = 2.0  # Jakarta Bay is microtidal; a model output beyond this means a bad point/config, not weather


def flag_rain(mm_day: pd.Series) -> pd.Series:
    """0=ok, 1=out of physical range (negative or implausibly high)."""
    bad = (mm_day < 0) | (mm_day > RAIN_MAX_PLAUSIBLE_MM_DAY)
    return bad.map({True: 1, False: 0}).astype("int8")


def flag_tide(tide_m: pd.Series) -> pd.Series:
    """0=ok, 1=implausible (model/config error, not real signal - EOT20 has no
    physical sensor noise to flag, only a wrong point/date would produce this)."""
    bad = tide_m.abs() > TIDE_MAX_PLAUSIBLE_M
    return bad.map({True: 1, False: 0}).astype("int8")
