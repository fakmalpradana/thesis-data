"""Siaga-3 exceedance F1/CSI, LSTM vs persistence, per station x horizon on the
LSTM test period; per-event (E2/E6/E7, the only events inside the test window)
detail with lead time and a BPBD/PetaBencana ground-truth cross-tab.

    python3 experiments/siaga_f1.py

Outputs: reports/siaga_f1/{metrics_station,metrics_event,event_crosstab}.csv, summary.md
"""
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "reports/lstm_multistation"
OUT = ROOT / "reports/siaga_f1"
STATIONS = [107, 126, 140, 150, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187]
HORIZONS = [1, 6, 12, 24]
# Only E2/E6/E7 fall inside the test period (2025-05-01+, 126: 2026-01-01+);
# E1/E3/E4/E5 are in train/val and cannot be scored from held-out predictions.
EVENTS = {"E2": ("2026-01-12", "2026-01-22"), "E6": ("2025-07-06", "2025-07-07"), "E7": ("2026-06-02", "2026-06-05")}
PAD = pd.Timedelta(days=1)  # window padding for both event scoring and the PetaBencana 3 km lookup


def confusion_metrics(obs, pred) -> dict:
    """TP/FP/FN -> F1, CSI, POD, FAR. F1/CSI are 0/0 (NaN) when TP+FP+FN==0, i.e.
    the station never exceeded the threshold AND the model never predicted it."""
    obs, pred = np.asarray(obs, bool), np.asarray(pred, bool)
    tp, fp, fn = int((obs & pred).sum()), int((~obs & pred).sum()), int((obs & ~pred).sum())
    denom = tp + fp + fn
    return dict(
        tp=tp, fp=fp, fn=fn,
        f1=2 * tp / (2 * tp + fp + fn) if denom else np.nan,
        csi=tp / denom if denom else np.nan,
        pod=tp / (tp + fn) if (tp + fn) else np.nan,
        far=fp / (tp + fp) if (tp + fp) else np.nan,
        n_obs_hours=int(obs.sum()),
    )


def self_check() -> None:
    obs = np.array([1, 0, 1, 1, 0, 0], bool)
    assert confusion_metrics(obs, obs)["f1"] == 1.0
    assert confusion_metrics(obs, ~obs)["f1"] == 0.0
    assert np.isnan(confusion_metrics(np.zeros(5, bool), np.zeros(5, bool))["f1"])


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bpbd_kejadian(con, kecamatan: str, year: int, month: int):
    """Sum jumlah_kejadian for a kecamatan in a given month, from whichever
    quarterly BPBD table+periode covers it. Returns (count, published: bool)."""
    periode = f"{year}{((month - 1) // 3 + 1) * 3:02d}"
    table = "bpbd_banjir_2025_2026" if year >= 2025 else "bpbd_banjir_2024"
    have_periode = con.sql(f"SELECT count(*) c FROM {table} WHERE periode_data = '{periode}'").df().c.iloc[0]
    if have_periode == 0:
        return np.nan, False  # not published yet (e.g. E7, June 2026)
    # kecamatan names are unique across all 5 DKI kota (verified against Kecamatan DKI.geojson),
    # so no wilayah filter is needed even for a station whose coords land outside Jakarta Utara.
    df = con.sql(
        f"SELECT jumlah_kejadian FROM {table} WHERE periode_data = '{periode}' AND bulan = '{month}' "
        f"AND kecamatan ILIKE '{kecamatan}'"
    ).df()
    return pd.to_numeric(df["jumlah_kejadian"], errors="coerce").sum(), True


def main() -> None:
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
    stations = con.sql(
        f"SELECT stasiun_id, nama, lat, lon, siaga2_cm, siaga3_cm FROM stations "
        f"WHERE stasiun_id IN ({','.join(map(str, STATIONS))})"
    ).df().set_index("stasiun_id")
    forcing = con.sql("SELECT waktu, stasiun_id, tma_mean FROM forcing_hourly_multi").df()
    forcing["waktu"] = pd.to_datetime(forcing["waktu"], utc=True).dt.tz_convert(None)
    forcing["stasiun_id"] = forcing["stasiun_id"].astype(int)

    preds = {}  # (sid, h) -> df with y, yhat, y_persist
    for sid in STATIONS:
        tma = forcing.loc[forcing.stasiun_id == sid].set_index("waktu")["tma_mean"]
        for h in HORIZONS:
            df = pd.read_parquet(PRED_DIR / f"pred_{sid}_h{h}.parquet")
            # persistence baseline = tma_mean at (target time - h), same y_last used in training
            df["y_persist"] = tma.reindex(df["waktu"] - pd.Timedelta(hours=h)).to_numpy()
            preds[(sid, h)] = df

    station_rows, event_rows = [], []
    for sid in STATIONS:
        thr2, thr3 = stations.loc[sid, "siaga2_cm"], stations.loc[sid, "siaga3_cm"]
        for h in HORIZONS:
            df = preds[(sid, h)]
            for level, thr in (("siaga2", thr2), ("siaga3", thr3)):
                obs = df["y"] >= thr
                for method, col in (("lstm", "yhat"), ("persistence", "y_persist")):
                    pred = df[col] >= thr
                    station_rows.append(dict(stasiun_id=sid, horizon=h, level=level, method=method,
                                              **confusion_metrics(obs, pred)))

        # per-event: siaga3 only (the task's binary event definition), + max lead time
        for eid, (estart, eend) in EVENTS.items():
            base = preds[(sid, 1)]
            win1 = (base["waktu"] >= pd.Timestamp(estart) - PAD) & (base["waktu"] <= pd.Timestamp(eend) + PAD)
            exceed = base.loc[win1 & (base["y"] >= thr3), "waktu"]
            t_obs = exceed.min() if len(exceed) else pd.NaT
            leads = []
            for h in HORIZONS:
                df = preds[(sid, h)]
                win = (df["waktu"] >= pd.Timestamp(estart) - PAD) & (df["waktu"] <= pd.Timestamp(eend) + PAD)
                sub = df[win]
                if sub.empty:
                    continue
                obs, predl, predp = sub["y"] >= thr3, sub["yhat"] >= thr3, sub["y_persist"] >= thr3
                for method, pred in (("lstm", predl), ("persistence", predp)):
                    event_rows.append(dict(event=eid, stasiun_id=sid, horizon=h, method=method,
                                            **confusion_metrics(obs, pred)))
                if pd.notna(t_obs):
                    row = df.loc[df["waktu"] == t_obs]
                    if len(row) and row["yhat"].iloc[0] >= thr3:
                        leads.append(h)  # yhat for this h-step-ahead forecast already flags t_obs
            max_lead = max(leads) if leads else (0 if pd.notna(t_obs) else np.nan)
            for r in event_rows:
                if r["event"] == eid and r["stasiun_id"] == sid:
                    r["t_first_obs_exceed"], r["max_lead_hours"] = t_obs, max_lead

    metrics_station = pd.DataFrame(station_rows)
    metrics_event = pd.DataFrame(event_rows)
    metrics_station.to_csv(OUT / "metrics_station.csv", index=False)
    metrics_event.to_csv(OUT / "metrics_event.csv", index=False)

    # --- event ground-truth cross-tab ---
    kec = gpd.read_file(ROOT / "reference/batas_adm/Kecamatan DKI.geojson")[["WADMKC", "geometry"]]
    have_xy = stations.dropna(subset=["lat", "lon"]).reset_index()
    pts = gpd.GeoDataFrame(have_xy[["stasiun_id"]],
                            geometry=gpd.points_from_xy(have_xy["lon"], have_xy["lat"]), crs=4326)
    joined = gpd.sjoin(pts, kec, predicate="within", how="left")[["stasiun_id", "WADMKC"]]
    kec_of = dict(zip(joined.stasiun_id, joined.WADMKC))
    # 170 has no lat/lon in `stations` (DSDA coords never resolved) -> no kecamatan/PetaBencana row possible
    # ponytail: nearest-station-name proxy would recover it; not worth it for 1 station of 14.

    pb = con.sql("SELECT created_at, lat, lon, flood_depth_cm FROM petabencana_reports WHERE disaster_type='flood'").df()
    pb["created_at"] = pd.to_datetime(pb["created_at"], utc=True).dt.tz_convert(None)

    crosstab_rows = []
    for eid, (estart, eend) in EVENTS.items():
        month = pd.Timestamp(estart).month
        for sid in STATIONS:
            df6 = preds[(sid, 6)]
            win6 = (df6["waktu"] >= pd.Timestamp(estart) - PAD) & (df6["waktu"] <= pd.Timestamp(eend) + PAD)
            df1 = preds[(sid, 1)]
            win1 = (df1["waktu"] >= pd.Timestamp(estart) - PAD) & (df1["waktu"] <= pd.Timestamp(eend) + PAD)
            thr3 = stations.loc[sid, "siaga3_cm"]
            obs_exceed = bool((df1.loc[win1, "y"] >= thr3).any()) if win1.any() else False
            pred_exceed_h6 = bool((df6.loc[win6, "yhat"] >= thr3).any()) if win6.any() else False
            kecamatan = kec_of.get(sid)
            if kecamatan is not None:
                bpbd_n, published = bpbd_kejadian(con, kecamatan, pd.Timestamp(estart).year, month)
            else:
                bpbd_n, published = np.nan, None
            if sid in have_xy["stasiun_id"].values:
                lat, lon = stations.loc[sid, "lat"], stations.loc[sid, "lon"]
                pwin = (pb["created_at"] >= pd.Timestamp(estart) - PAD) & (pb["created_at"] <= pd.Timestamp(eend) + PAD)
                d = haversine_km(lat, lon, pb.loc[pwin, "lat"], pb.loc[pwin, "lon"])
                near = pb.loc[pwin][d <= 3]
                pb_reports, pb_depth = len(near), near["flood_depth_cm"].median()
            else:
                pb_reports, pb_depth = np.nan, np.nan
            crosstab_rows.append(dict(
                event=eid, stasiun_id=sid, kecamatan=kecamatan, obs_exceed=obs_exceed,
                pred_exceed_h6=pred_exceed_h6,
                bpbd_kejadian_bulan=bpbd_n if published is not False else "not_published",
                pb_reports=pb_reports, pb_depth_median_cm=pb_depth,
            ))
    event_crosstab = pd.DataFrame(crosstab_rows)
    event_crosstab.to_csv(OUT / "event_crosstab.csv", index=False)

    # --- summary.md ---
    piv = (metrics_station[metrics_station.level == "siaga3"]
           .groupby(["horizon", "method"])[["f1", "csi"]].median().round(3))
    undefined = (metrics_station[(metrics_station.level == "siaga3") & (metrics_station.method == "lstm")
                                  & metrics_station.f1.isna()]["stasiun_id"].unique())
    lines = [
        "# Siaga-3 exceedance F1/CSI — summary\n",
        "Median F1/CSI across the 14 stations, per horizon, LSTM vs persistence (test period, Siaga 3):\n",
        "```", piv.to_string(), "```\n",
        f"**F1 undefined** (NaN) for stations {sorted(int(s) for s in undefined)}: they never cross "
        "Siaga 3 in the LSTM test window (TP=FP=FN=0), so F1/CSI are 0/0.\n" if len(undefined) else
        "All 14 stations exceed Siaga 3 at least once in the test period; F1 is defined everywhere.\n",
        "## Events vs ground truth\n",
    ]
    for eid in EVENTS:
        sub = event_crosstab[event_crosstab.event == eid]
        seen = sub.loc[sub.obs_exceed, "stasiun_id"].tolist()
        pb_hit = int(sub["pb_reports"].fillna(0).sum())
        bpbd_vals = sub["bpbd_kejadian_bulan"].dropna()
        bpbd_vals = bpbd_vals[bpbd_vals != "not_published"]
        bpbd_txt = f"BPBD kejadian (kecamatan-month, summed over stations' kecamatan) {bpbd_vals.sum():.0f}" \
            if len(bpbd_vals) else "BPBD not yet published for this month"
        lines.append(f"- **{eid}**: TMA network saw exceedance at stations {seen or 'none'}; "
                      f"{bpbd_txt}; {pb_hit} PetaBencana reports within 3 km across stations.\n")
    (OUT / "summary.md").write_text("\n".join(lines))

    print(piv.unstack("method"))
    print(f"\nF1 undefined at stations: {sorted(int(s) for s in undefined)}")
    print(f"\nWrote {OUT}/metrics_station.csv, metrics_event.csv, event_crosstab.csv, summary.md")


if __name__ == "__main__":
    main()
