"""Cross-correlation (tide/rain -> TMA) per station and per flood event.

    marimo edit notebooks/03_xcorr_events.py

Whole-record lag scan (0-72h) per station for tide and rain, plus a per-event
(E1-E7, observations only - no LSTM needed) peak/lag table used to classify
each event as pluvial/tidal/compound and compare against `validation/events.csv`.

Outputs: reports/xcorr/{xcorr_lags,event_response}.csv
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "reports" / "xcorr"
    OUT.mkdir(parents=True, exist_ok=True)
    STATIONS = [107, 126, 140, 150, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187]
    MAX_LAG = 72

    con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
    forcing = con.sql(
        "SELECT waktu, stasiun_id, stasiun_nama, tma_mean, hujan_mm, tide_m FROM forcing_hourly_multi"
    ).df()
    forcing["waktu"] = pd.to_datetime(forcing["waktu"], utc=True).dt.tz_convert(None)
    forcing["stasiun_id"] = forcing["stasiun_id"].astype(int)
    events = con.sql("SELECT * FROM events").df()
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])
    return MAX_LAG, OUT, ROOT, STATIONS, con, events, forcing, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Cross-correlation: tide & rain -> TMA, per station and per event

    Two analyses on **observations** (`forcing_hourly_multi`), not model predictions, so all
    7 curated events (`validation/events.csv`) are in scope, not just the 3 in the LSTM test
    window. Rain (CHIRPS) is daily-constant, so a lagged rain correlation only resolves to
    ~24h steps within a day — reported anyway, flagged as coarse.

    1. **Whole-record lag scan** (0-72h): `corr(TMA(t), driver(t-lag))` per station, for
       `tide_m` and `hujan_mm`, best |corr| lag reported.
    2. **Per-event** (window = start-3d .. end+3d): peak TMA, tide at peak, 72h antecedent
       rain, lag from peak tide to peak TMA, lag from rain onset to peak TMA -> classified
       pluvial / tidal / compound, compared with the event's `type` in `events.csv`.
    """)
    return


@app.cell
def _(MAX_LAG, np, pd):
    def xcorr_curve(target: pd.Series, driver: pd.Series, max_lag: int = MAX_LAG):
        """corr(target(t), driver(t-lag)) for lag in [0, max_lag]. Pairwise NaN-safe."""
        out = []
        for lag in range(max_lag + 1):
            shifted = driver.shift(lag)
            mask = target.notna() & shifted.notna()
            if mask.sum() < 30:
                out.append((lag, np.nan))
                continue
            out.append((lag, np.corrcoef(target[mask], shifted[mask])[0, 1]))
        return pd.DataFrame(out, columns=["lag_h", "corr"])

    def best_lag(curve: pd.DataFrame) -> tuple[int, float]:
        i = curve["corr"].abs().idxmax()
        return int(curve.loc[i, "lag_h"]), float(curve.loc[i, "corr"])
    return best_lag, xcorr_curve


@app.cell
def _(STATIONS, best_lag, forcing, mo, xcorr_curve):
    curves = {}  # sid -> {"tide": df, "rain": df}
    best_rows = []
    for _sid in STATIONS:
        _g = forcing.loc[forcing.stasiun_id == _sid].set_index("waktu").sort_index()
        _c_tide = xcorr_curve(_g["tma_mean"], _g["tide_m"])
        _c_rain = xcorr_curve(_g["tma_mean"], _g["hujan_mm"])
        curves[_sid] = {"tide": _c_tide, "rain": _c_rain}
        for _var, _c in (("tide", _c_tide), ("rain", _c_rain)):
            _lag, _corr = best_lag(_c)
            best_rows.append({"stasiun_id": _sid, "var": _var, "lag_h": _lag, "corr": round(_corr, 3)})

    import pandas as _pd
    xcorr_lags = _pd.DataFrame(best_rows)
    pick_station = mo.ui.dropdown([str(s) for s in STATIONS], value=str(STATIONS[2]), label="station (140 default)")
    mo.vstack([mo.md("## 1. Whole-record cross-correlation — best lag per station"),
               mo.ui.table(xcorr_lags.pivot(index="stasiun_id", columns="var", values=["lag_h", "corr"]),
                            selection=None),
               pick_station])
    return curves, pick_station, xcorr_lags


@app.cell
def _(curves, mo, pick_station, plt):
    _sid = int(pick_station.value)
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _ax[0].plot(curves[_sid]["tide"].lag_h, curves[_sid]["tide"]["corr"], color="tab:green")
    _ax[0].set_title(f"station {_sid}: corr(TMA(t), tide(t-lag))"); _ax[0].set_xlabel("lag (h)")
    _ax[1].plot(curves[_sid]["rain"].lag_h, curves[_sid]["rain"]["corr"], color="tab:blue")
    _ax[1].set_title(f"station {_sid}: corr(TMA(t), rain(t-lag))"); _ax[1].set_xlabel("lag (h)")
    mo.vstack([_fig])
    return


@app.cell
def _(mo, np, pd):
    def peak_stats(g: pd.DataFrame):
        """g: hourly rows (waktu index) for one station within an event window."""
        if g["tma_mean"].notna().sum() == 0:
            return None
        peak_t = g["tma_mean"].idxmax()
        peak_v = g["tma_mean"].max()
        tide_at_peak = g["tide_m"].get(peak_t, np.nan)
        pre72 = g.loc[peak_t - pd.Timedelta(hours=72):peak_t]
        rain_cum_72h = pre72["hujan_mm"].sum(min_count=1) / 24  # hourly rows repeat the daily CHIRPS value 24x -> /24 recovers true mm over the window
        tide_max_t = g["tide_m"].idxmax() if g["tide_m"].notna().any() else pd.NaT
        lag_tide_h = (peak_t - tide_max_t) / pd.Timedelta(hours=1) if pd.notna(tide_max_t) else np.nan
        onset = g.loc[g["hujan_mm"] > 0]
        rain_onset_t = onset.index.min() if len(onset) else pd.NaT
        lag_rain_h = (peak_t - rain_onset_t) / pd.Timedelta(hours=1) if pd.notna(rain_onset_t) else np.nan
        return dict(peak_tma_cm=round(peak_v, 1), peak_time=peak_t, tide_at_peak_m=round(tide_at_peak, 3)
                    if pd.notna(tide_at_peak) else np.nan, rain_cum_72h_mm=rain_cum_72h,
                    lag_tide_to_peak_h=lag_tide_h, lag_rain_onset_to_peak_h=lag_rain_h)


    def classify(rain_cum_72h, lag_tide_h) -> str:
        """Ponytail heuristic (no numeric thresholds were specified in the plan): pure-tide
        events have negligible antecedent rain and a short peak-tide-to-peak-TMA lag; pluvial
        events have meaningful rain and a longer/absent tide lag; anything with both signals
        present is compound. Thresholds (5 mm, 6 h) are a reasonable first cut, not tuned.
        """
        has_rain = pd.notna(rain_cum_72h) and rain_cum_72h >= 5
        tide_led = pd.notna(lag_tide_h) and abs(lag_tide_h) <= 6
        if has_rain and tide_led:
            return "compound"
        if has_rain:
            return "pluvial"
        if tide_led:
            return "tidal"
        return "compound"  # neither clearly dominant -> conservative default
    return classify, peak_stats


@app.cell
def _(STATIONS, classify, events, forcing, mo, peak_stats, pd):
    PAD3 = pd.Timedelta(days=3)
    resp_rows = []
    for _, _ev in events.iterrows():
        _win_lo, _win_hi = _ev["start"] - PAD3, _ev["end"] + PAD3
        _station_stats = []
        for _sid in STATIONS:
            _g = forcing.loc[(forcing.stasiun_id == _sid) & (forcing.waktu >= _win_lo) & (forcing.waktu <= _win_hi)]
            if _g.empty:
                continue
            _g = _g.set_index("waktu").sort_index()
            _s = peak_stats(_g)
            if _s is not None:
                _s.update(event=_ev["event_id"], stasiun_id=_sid)
                _station_stats.append(_s)
        if not _station_stats:
            continue
        _df = pd.DataFrame(_station_stats)
        _rain_med = _df["rain_cum_72h_mm"].median()
        _tide_lag_med = _df["lag_tide_to_peak_h"].median()
        _type_computed = classify(_rain_med, _tide_lag_med)
        _df["event_type_given"] = _ev["type"]
        _df["event_type_computed"] = _type_computed
        resp_rows.append(_df)

    event_response = pd.concat(resp_rows, ignore_index=True)
    class_summary = (event_response[["event", "event_type_given", "event_type_computed"]]
                      .drop_duplicates().reset_index(drop=True))
    class_summary["agree"] = class_summary["event_type_given"].str.contains(
        class_summary["event_type_computed"], case=False, regex=False
    ) if False else [g.lower().find(c) >= 0 for g, c in
                      zip(class_summary["event_type_given"], class_summary["event_type_computed"])]
    pick_event = mo.ui.dropdown(class_summary["event"].tolist(), value=class_summary["event"].iloc[0], label="event")
    mo.vstack([mo.md("## 2. Per-event peak/lag table and classification"),
               mo.ui.table(class_summary, selection=None),
               mo.ui.table(event_response.round(2), selection=None, page_size=15),
               pick_event])
    return class_summary, event_response, pick_event


@app.cell
def _(STATIONS, events, forcing, mo, pd, pick_event, plt):
    _ev = events.loc[events.event_id == pick_event.value].iloc[0]
    _win_lo, _win_hi = _ev["start"] - pd.Timedelta(days=3), _ev["end"] + pd.Timedelta(days=3)
    _fig, _ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for _sid in STATIONS:
        _g = forcing.loc[(forcing.stasiun_id == _sid) & (forcing.waktu >= _win_lo) & (forcing.waktu <= _win_hi)]
        if _g.empty or _g["tma_mean"].notna().sum() == 0:
            continue
        _norm = _g["tma_mean"] - _g["tma_mean"].mean()
        _ax[0].plot(_g["waktu"], _norm, lw=0.6, label=str(_sid))
        _pt = _g.set_index("waktu")["tma_mean"].idxmax()
        _ax[0].scatter([_pt], [_norm[_g["waktu"] == _pt].iloc[0]], s=10)
    _ax[0].set_ylabel("TMA, mean-removed (cm)"); _ax[0].legend(fontsize=6, ncol=7)
    _r = forcing.loc[(forcing.stasiun_id == STATIONS[0]) & (forcing.waktu >= _win_lo) & (forcing.waktu <= _win_hi)]
    _ax[1].bar(_r["waktu"], _r["hujan_mm"], width=0.04, color="tab:blue"); _ax[1].set_ylabel("rain (mm/day)")
    _ax[2].plot(_r["waktu"], _r["tide_m"], lw=0.6, color="tab:green"); _ax[2].set_ylabel("tide (m)")
    _fig.suptitle(f"{pick_event.value}: {_ev['start'].date()} -> {_ev['end'].date()} ({_ev['type']})")
    mo.vstack([_fig])
    return


@app.cell
def _(OUT, class_summary, event_response, mo, xcorr_lags):
    xcorr_lags.to_csv(OUT / "xcorr_lags.csv", index=False)
    event_response.to_csv(OUT / "event_response.csv", index=False)
    _n_agree = int(class_summary["agree"].sum())
    _disagree = class_summary.loc[~class_summary["agree"], "event"].tolist()
    mo.md(f"""
    ## 3. Summary

    Wrote `reports/xcorr/xcorr_lags.csv` ({len(xcorr_lags)} rows) and
    `reports/xcorr/event_response.csv` ({len(event_response)} rows).

    Classification agrees with `events.csv` `type` for **{_n_agree}/{len(class_summary)}**
    events; disagreements: **{_disagree or 'none'}**. The classifier is a simple two-threshold
    heuristic (median 72h antecedent rain >= 5 mm, median |peak-tide-to-peak-TMA lag| <= 6h)
    since the plan did not fix exact triggers — treat disagreements as a prompt to retune
    thresholds against the curated `notes` column, not as ground truth being wrong.
    """)
    return


if __name__ == "__main__":
    app.run()
