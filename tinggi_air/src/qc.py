"""Quality control: flag, never silently drop or fill (brief §4.3).

qc_flag (point-level):
  0 = pass
  1 = out of physical range (-999 sentinel, or beyond p0.1/p99.9 calibrated
      per station from its own data). NOTE: negative/zero values are NOT
      flagged by themselves - papan duga datums are per-station and
      arbitrary (brief §7); some stations read consistently negative under
      normal conditions (e.g. Rumah Pompa Pluit, datum above typical water
      level). Only the percentile check and the explicit sentinel catch
      genuine anomalies here.
  2 = stuck sensor (identical value for longer than a fixed 60-minute
      duration - NOT a fixed row count. The brief's "6 pengamatan" was
      calibrated against its one worked example (~10min interval, i.e.
      ~1 hour); interval varies wildly across stations (~1 to ~11 min, see
      DISCOVERY.md §6), so "6 rows" means a different real duration per
      station - and scaling the threshold BY each station's own interval
      (tried first) makes it worse, not better: a 1-min station would then
      get flagged "stuck" after just 6 minutes of ordinary calm water. A
      fixed absolute duration, applied uniformly, is what "6 obs @ 10min"
      actually meant.)
  3 = unphysical spike (rate of change beyond a per-station 99.9th-percentile
      calibrated threshold, normalized to cm/10min to tolerate gaps)
  4 = duplicate timestamp (all copies after the first)
Priority when more than one applies: 1 > 4 > 2 > 3 > 0 (physical impossibility
outranks everything else).

qc_flag (hourly, extension beyond the point-level scale):
  0 = ok (n_obs >= 2, and most of this hour's raw points were point-flag 0)
  8 = degraded (n_obs >= 2 but >=50% of this hour's raw points carried a
      non-zero point-level qc_flag - tma_mean/tma_max are still computed
      from all of them, never silently dropped, but this says so explicitly
      instead of looking identical to a clean hour. See resample_hourly()
      docstring for why this was added.)
  9 = gap (n_obs < 2) - do not silently treat as valid.
See also the `pct_flagged` column (exact fraction, not just the 0.5 cutoff).
"""
from __future__ import annotations

import pandas as pd

MIN_OBS_FOR_PERCENTILE = 100
STUCK_DURATION = pd.Timedelta(minutes=60)  # ~ brief's "6 pengamatan" @ its ~10min worked example
SENTINELS = {-999.0}


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    """df: waktu, stasiun_id, stasiun_nama, tma_cm (single station, already sorted or not).
    Returns a copy with an added int8 qc_flag column."""
    df = df.sort_values("waktu").reset_index(drop=True).copy()
    n = len(df)
    flag = pd.Series(0, index=df.index, dtype="int16")  # widen during computation, cast to int8 at the end

    # 1. physical range - only the explicit sentinel, never a bare "<=0" rule
    # (per-station datums are arbitrary, see module docstring)
    is_sentinel = df["tma_cm"].isin(SENTINELS)
    if n >= MIN_OBS_FOR_PERCENTILE:
        lo, hi = df["tma_cm"].quantile([0.001, 0.999])
        out_of_range = (df["tma_cm"] < lo) | (df["tma_cm"] > hi)
    else:
        out_of_range = pd.Series(False, index=df.index)
    flag_range = is_sentinel | out_of_range
    flag = flag.mask(flag_range, 1)

    # 2. duplicate timestamp - all copies after the first
    is_dup = df["waktu"].duplicated(keep="first")
    flag = flag.mask(is_dup & (flag == 0), 4)

    # 3. stuck sensor - identical value for longer than STUCK_DURATION,
    # measured as wall-clock time (not row count - see module docstring).
    same_as_prev = df["tma_cm"].eq(df["tma_cm"].shift())
    run_id = (~same_as_prev).cumsum()
    run_span = df.groupby(run_id)["waktu"].transform(lambda s: s.max() - s.min())
    is_stuck = run_span > STUCK_DURATION
    flag = flag.mask(is_stuck & (flag == 0), 2)

    # 4. unphysical spike - rate of change normalized to cm/10min, threshold
    # calibrated from this station's own data (99.9th pct of |rate|), never guessed.
    minutes = df["waktu"].diff().dt.total_seconds() / 60.0
    rate = (df["tma_cm"].diff().abs() / minutes.replace(0, pd.NA)) * 10.0
    if rate.notna().sum() >= MIN_OBS_FOR_PERCENTILE:
        spike_threshold = rate.quantile(0.999)
        is_spike = rate > spike_threshold
        flag = flag.mask(is_spike.fillna(False) & (flag == 0), 3)

    df["qc_flag"] = flag.astype("int8")
    return df


DEGRADED_FRACTION_THRESHOLD = 0.5  # majority of an hour's raw points point-flagged -> hourly qc_flag=8


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station hourly aggregation. tma_max kept explicitly - a mean-only
    aggregate would shave off the flood peaks that are the whole point (brief §4.2).

    Point-level qc_flag propagation (added after a real bug was found: TMA
    vs an independent EOT20 tide-model cross-check showed correlation
    decaying from 0.92 (2021) to 0.34 (2026) at station 140, tracking a jump
    in point-level stuck-sensor flags from ~2% to 22-27%/year from 2024
    onward - none of that was visible at the hourly level, which only ever
    tracked n_obs<2 gaps. tma_mean/tma_max themselves still include flagged
    points (never silently drop, brief's own QC principle) - but now the
    fraction is explicit, not hidden:
      qc_flag: 0=ok, 8=degraded (n_obs>=2 but most raw points this hour were
      point-flagged - i.e. an average/max computed mostly from stuck/spike/
      out-of-range readings), 9=gap (n_obs<2, unchanged).
      pct_flagged: exact fraction of this hour's raw points with point-level
      qc_flag != 0, for anyone who wants a finer cutoff than the 0.5 default."""
    out = []
    for (stasiun_id, stasiun_nama), g in df.groupby(["stasiun_id", "stasiun_nama"]):
        g = g.set_index("waktu").sort_index()
        hourly = g["tma_cm"].resample("1h").agg(tma_mean="mean", tma_max="max", n_obs="count")
        pct_flagged = (g["qc_flag"] != 0).resample("1h").mean()
        hourly["pct_flagged"] = pct_flagged.reindex(hourly.index).fillna(0.0).round(4)

        is_gap = hourly["n_obs"] < 2
        is_degraded = (~is_gap) & (hourly["pct_flagged"] >= DEGRADED_FRACTION_THRESHOLD)
        hourly["qc_flag"] = 0
        hourly.loc[is_degraded, "qc_flag"] = 8
        hourly.loc[is_gap, "qc_flag"] = 9
        hourly["qc_flag"] = hourly["qc_flag"].astype("int8")

        hourly["stasiun_id"] = stasiun_id
        hourly["stasiun_nama"] = stasiun_nama
        out.append(hourly.reset_index())
    if not out:
        return pd.DataFrame(
            columns=["waktu", "tma_mean", "tma_max", "n_obs", "pct_flagged", "qc_flag", "stasiun_id", "stasiun_nama"]
        )
    return pd.concat(out, ignore_index=True)
