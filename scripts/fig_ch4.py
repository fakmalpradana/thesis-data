"""Export Chapter 4 figures (4.1-4.7) as PNG (200 dpi) + SVG.

Run from repo root: python3 scripts/fig_ch4.py
Reads catalog.duckdb read_only where possible, falls back to parquet/gpkg
files directly if the db is locked by a marimo session.

ponytail: one script, one file per figure function, no config/plugin system
for what is a fixed list of 7 figures that will not grow.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures" / "ch4"
OUT.mkdir(parents=True, exist_ok=True)

MM_PER_IN = 25.4
WIDE = 160 / MM_PER_IN
NARROW = 80 / MM_PER_IN
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                      "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
TAB10 = plt.get_cmap("tab10").colors

NOTES = []  # (fig_id, note) collected for the README/report


def save(fig, name):
    png = OUT / f"{name}.png"
    svg = OUT / f"{name}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    for p in (png, svg):
        assert p.stat().st_size < 2 * 1024 * 1024, f"{p} exceeds 2 MB"


def try_duckdb():
    """Return a read-only duckdb connection, or None if locked."""
    try:
        con = duckdb.connect(str(ROOT / "catalog.duckdb"), read_only=True)
        con.execute("select 1").fetchall()
        return con
    except Exception as e:
        NOTES.append(("*", f"catalog.duckdb unavailable ({e.__class__.__name__}), read parquet/gpkg directly"))
        return None


# ---------------------------------------------------------------- data ----

def load_140_hourly():
    files = sorted((ROOT / "tinggi_air/data/processed").glob("140_*_hourly.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["waktu"] = df["waktu"].dt.tz_localize(None)
    return df.sort_values("waktu")


def load_pred(report_dir, config_or_station, horizon):
    return pd.read_parquet(ROOT / "reports" / report_dir / f"pred_{config_or_station}_h{horizon}.parquet")


def load_forcing_gsmap():
    df = pd.read_parquet(ROOT / "forcing-acquisition/data/processed/forcing_hourly_multi_gsmap.parquet")
    df["waktu"] = df["waktu"].dt.tz_localize(None)
    return df


# ------------------------------------------------------------ figure 4.1 --

def fig_4_1():
    df = load_140_hourly()
    fig, axes = plt.subplots(2, 1, figsize=(WIDE, 3.6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(df["waktu"], df["tma_mean"], lw=0.3, color=TAB10[0])
    # B_clean split bands: train 2021-2023, val 2024, test 2025-2026 (calendar-year
    # approximation; ponytail: exact split drops flagged hours inside each band,
    # not recoverable from saved artifacts, so bands are the nominal calendar years).
    bands = [("2021-01-01", "2024-01-01", "train", "#ddeeff"),
             ("2024-01-01", "2025-01-01", "val", "#fff2cc"),
             ("2025-01-01", "2026-09-08", "test", "#e6ffe6")]
    for start, end, label, color in bands:
        for a in axes:
            a.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.6, lw=0)
    ax.annotate("2024 onset of degradation", xy=(pd.Timestamp("2024-01-01"), df["tma_mean"].max()),
                xytext=(pd.Timestamp("2024-06-01"), df["tma_mean"].max()),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.set_ylabel("TMA (cm)")
    ax2 = axes[1]
    monthly = df.set_index("waktu")["qc_flag"].eq(8).resample("MS").mean() * 100
    ax2.plot(monthly.index, monthly.values, color=TAB10[3], lw=1)
    ax2.set_ylabel("flagged (%)")
    ax2.set_xlabel("year")
    fig.align_ylabels(axes)
    save(fig, "fig4_1_station140_tma_flagged")


# ------------------------------------------------------------ figure 4.2 --

def fig_4_2():
    df = load_pred("lstm_baseline", "B_clean", 6)
    fig, axes = plt.subplots(2, 1, figsize=(WIDE, 3.6))
    ax = axes[0]
    ax.plot(df["waktu"], df["y"], lw=0.5, color="k", label="observed")
    ax.plot(df["waktu"], df["yhat"], lw=0.5, color=TAB10[1], label="predicted (h=6)")
    ax.set_ylabel("TMA (cm)")
    ax.legend(loc="upper right", frameon=False)

    peak_t = df.loc[df["y"].idxmax(), "waktu"]
    zoom_start = peak_t - pd.Timedelta(days=15)
    zoom_end = peak_t + pd.Timedelta(days=15)
    zoom = df[(df["waktu"] >= zoom_start) & (df["waktu"] <= zoom_end)]
    ax2 = axes[1]
    ax2.plot(zoom["waktu"], zoom["y"], lw=0.8, color="k", label="observed")
    ax2.plot(zoom["waktu"], zoom["yhat"], lw=0.8, color=TAB10[1], label="predicted (h=6)")
    ax2.set_ylabel("TMA (cm)")
    ax2.set_xlabel("30-day zoom around largest observed peak")
    fig.autofmt_xdate()
    save(fig, "fig4_2_bclean_h6_obs_vs_pred")


# ------------------------------------------------------------ figure 4.3 --

def fig_4_3():
    with open(ROOT / "reports/lstm_baseline/metrics.json") as f:
        metrics = pd.DataFrame(json.load(f))
    horizons = sorted(metrics["horizon"].unique())
    fig, ax = plt.subplots(figsize=(NARROW, NARROW))
    configs = ["A_chrono", "B_clean", "C_pre2024"]
    for i, cfg in enumerate(configs):
        sub = metrics[metrics["config"] == cfg].sort_values("horizon")
        ax.plot(sub["horizon"], sub["lstm_nse"], color=TAB10[i], marker="o", ms=3, label=cfg)
        ax.plot(sub["horizon"], sub["lstm_nse_flag0"], color=TAB10[i], marker="o", ms=3, ls="--")
    # persistence: same across flag-status split by construction for B_clean; show B_clean's as
    # representative grey reference (A_chrono/C_pre2024 persistence differ only slightly).
    sub = metrics[metrics["config"] == "B_clean"].sort_values("horizon")
    ax.plot(sub["horizon"], sub["persistence_nse"], color="0.5", marker="x", ms=4, label="persistence")
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("NSE")
    ax.set_xticks(horizons)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig4_3_nse_vs_horizon_configs")
    NOTES.append(("4.3", "persistence line uses B_clean persistence NSE only (grey, representative) "
                          "-- A_chrono/C_pre2024 persistence not overplotted to keep the legend readable"))


# ------------------------------------------------------------ figure 4.4 --

def fig_4_4():
    df = pd.read_csv(ROOT / "reports/compare_rain_loss/metrics.csv")
    order = ["chirps_mse", "gsmap_mse", "gsmap_quantile", "gsmap_mse_persist", "gsmap_mse_persist_resid"]
    legend_names = {"chirps_mse": "CHIRPS · MSE", "gsmap_mse": "GSMaP · MSE",
                     "gsmap_quantile": "GSMaP · Quantile", "gsmap_mse_persist": "R1 (GSMaP+persist)",
                     "gsmap_mse_persist_resid": "R2 (GSMaP+persist, residual)"}
    fig, ax = plt.subplots(figsize=(NARROW, NARROW))
    med_persist = df.groupby("horizon")["persistence_nse"].median().sort_index()
    ax.plot(med_persist.index, med_persist.values, color="0.5", marker="x", ms=4, label="persistence")
    for i, cfg in enumerate(order):
        sub = df[df["config"] == cfg]
        med = sub.groupby("horizon")["lstm_nse"].median().sort_index()
        ax.plot(med.index, med.values, color=TAB10[i], marker="o", ms=3, label=legend_names[cfg])
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("median NSE")
    ax.set_xticks(sorted(df["horizon"].unique()))
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "fig4_4_median_nse_vs_horizon_configs")


# ------------------------------------------------------------ figure 4.5 --

def fig_4_5():
    t0, t1 = pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-24")
    forcing = load_forcing_gsmap()
    fig, axes = plt.subplots(4, 1, figsize=(WIDE, 6.0), sharex=True,
                              gridspec_kw={"height_ratios": [1.4, 1.4, 1, 1]})

    for ax, sid, report_station in ((axes[0], "140", "140"), (axes[1], "107", "107")):
        pred = load_pred("lstm_multistation_gsmap_mse_persist_resid", report_station, 6)
        pred = pred[(pred["waktu"] >= t0) & (pred["waktu"] <= t1)]
        ax.plot(pred["waktu"], pred["y"], lw=0.8, color="k", label="observed")
        ax.plot(pred["waktu"], pred["yhat"], lw=0.8, color=TAB10[1], label="R2 forecast (h=6)")
        ax.set_ylabel(f"TMA {sid} (cm)")
        ax.legend(frameon=False, loc="upper right", fontsize=7)

    rain = forcing[(forcing["stasiun_id"] == "140") & (forcing["waktu"] >= t0) & (forcing["waktu"] <= t1)]
    axes[2].bar(rain["waktu"], rain["hujan_mm"], width=1 / 24, color=TAB10[0])
    axes[2].set_ylabel("GSMaP rain\n(mm h⁻¹)")

    tide = forcing[(forcing["stasiun_id"] == "140") & (forcing["waktu"] >= t0) & (forcing["waktu"] <= t1)]
    axes[3].plot(tide["waktu"], tide["tide_m"], lw=0.8, color=TAB10[2])
    axes[3].set_ylabel("EOT20 tide (m)")
    axes[3].set_xlabel("10–24 Jan 2026 (event E2)")
    fig.autofmt_xdate()
    save(fig, "fig4_5_event_E2_140_107")


# ------------------------------------------------------------ figure 4.6 --

def fig_4_6():
    df = pd.read_csv(ROOT / "reports/compare_rain_loss/metrics.csv")
    sub = df[(df["horizon"] == 6) & df["config"].isin(["gsmap_mse", "gsmap_quantile", "gsmap_mse_persist_resid"])].dropna(subset=["f1_p95"])
    markers = {"gsmap_mse": "o", "gsmap_quantile": "s", "gsmap_mse_persist_resid": "^"}
    legend_names = {"gsmap_mse": "GSMaP · MSE", "gsmap_quantile": "GSMaP · Quantile",
                     "gsmap_mse_persist_resid": "R2 (GSMaP+persist, residual)"}
    bubble_stations = {"179", "181", "184", "187"}
    fig, ax = plt.subplots(figsize=(NARROW, NARROW))
    for cfg, marker in markers.items():
        s = sub[sub["config"] == cfg]
        ax.scatter(s["far_p95"], s["f1_p95"], marker=marker, s=18, color=TAB10[0],
                   facecolors="none" if cfg != "gsmap_mse_persist_resid" else TAB10[0],
                   label=legend_names[cfg])
    for _, row in sub.iterrows():
        if str(row["stasiun_id"]) in bubble_stations:
            ax.annotate(str(row["stasiun_id"]), (row["far_p95"], row["f1_p95"]), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("FAR (p95, h=6)")
    ax.set_ylabel("F1 (p95, h=6)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig4_6_f1_far_scatter")


# ------------------------------------------------------------ figure 4.7 --

def fig_4_7():
    import geopandas as gpd

    boundary = gpd.read_file(ROOT / "reference/batas_adm/Batas Kota DKI.geojson")
    boundary = boundary[boundary["NAMOBJ"] == "Kota Adm. Jakarta Utara"]

    edges = gpd.read_file(ROOT / "reference/jakarta/canal_graph.gpkg", layer="canal_graph")
    boundary = boundary.to_crs(edges.crs)

    con = try_duckdb()
    if con is not None:
        stations = con.execute("select * from stations").df()
        con.close()
    else:
        import yaml
        with open(ROOT / "tinggi_air/config/stations.yaml") as f:
            cfg = yaml.safe_load(f)["stations"]
        stations = pd.DataFrame([{"stasiun_id": s["id"], "nama": s["nama"],
                                   "lat": s.get("koordinat", {}).get("lat"),
                                   "lon": s.get("koordinat", {}).get("lon")} for s in cfg])

    canal_ids = {107, 126, 140, 150, 162, 164, 166, 167, 169, 170}
    bubble_ids = {179, 181, 184, 187}
    isolated_ids = {164}
    outlet_ids = {140}
    stations = stations[stations["stasiun_id"].astype(int).isin(canal_ids | bubble_ids)].copy()
    stations["stasiun_id"] = stations["stasiun_id"].astype(int)
    stations = stations.dropna(subset=["lat", "lon"])

    def classify(sid):
        if sid in outlet_ids:
            return "outlet"
        if sid in isolated_ids:
            return "isolated"
        if sid in bubble_ids:
            return "bubble pump"
        return "canal"

    stations["cls"] = stations["stasiun_id"].apply(classify)
    gdf = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
                            crs="EPSG:4326").to_crs(edges.crs)

    # view extent: Jakarta Utara boundary + the 14 stations (some, e.g. 150, sit
    # upstream outside the admin boundary but must stay visible), not the full
    # upstream watershed the canal graph carries (it reaches into the Bogor hills).
    bminx, bminy, bmaxx, bmaxy = boundary.total_bounds
    sminx, sminy, smaxx, smaxy = gdf.total_bounds
    minx, miny = min(bminx, sminx), min(bminy, sminy)
    maxx, maxy = max(bmaxx, smaxx), max(bmaxy, smaxy)
    buf = 2000
    xlim, ylim = (minx - buf, maxx + buf), (miny - buf, maxy + buf)
    edges = edges.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]

    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fig, ax = plt.subplots(figsize=(WIDE, WIDE * 0.95))
    boundary.boundary.plot(ax=ax, color="black", lw=0.8)
    low_z = edges[edges["z_mean"] < 0]
    high_z = edges[~(edges["z_mean"] < 0)]
    # ponytail: 68k edges as vector paths blow the 2 MB SVG cap; rasterize just
    # this layer (rest of the figure -- points, labels, boundary -- stays vector).
    high_z.plot(ax=ax, color="0.75", lw=0.4, rasterized=True)
    low_z.plot(ax=ax, color=TAB10[0], lw=0.5, rasterized=True)

    # plot order matters: 140/166 share one coordinate pair (stations.yaml), so
    # draw "outlet" last to keep it from being hidden under the "canal" dot.
    cls_colors = {"canal": TAB10[1], "bubble pump": TAB10[2], "isolated": TAB10[4], "outlet": TAB10[3]}
    for cls, color in cls_colors.items():
        sub = gdf[gdf["cls"] == cls]
        ax.scatter(sub.geometry.x, sub.geometry.y, color=color, s=22, label=cls, zorder=5, edgecolor="k", lw=0.3)
    label_off = {166: (3, -9), 170: (6, 12), 140: (-18, 3), 162: (-14, -10), 184: (3, -9), 126: (-16, 4), 187: (-6, 5)}  # de-overlap
    for _, row in gdf.iterrows():
        ax.annotate(str(row["stasiun_id"]), (row.geometry.x, row.geometry.y), fontsize=6,
                    xytext=label_off.get(int(row["stasiun_id"]), (3, 3)), textcoords="offset points")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    # --- UTM grid in metres, northing labels rotated; no geographic axes
    m_fmt = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.xaxis.set_major_locator(MultipleLocator(5000)); ax.yaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(m_fmt); ax.yaxis.set_major_formatter(m_fmt)
    ax.tick_params(axis="y", labelrotation=90, labelsize=7); ax.tick_params(axis="x", labelsize=7)
    for lab in ax.get_yticklabels():
        lab.set_va("center")
    ax.set_xlabel("Easting (m) — WGS 84 / UTM zone 48S"); ax.set_ylabel("Northing (m) — WGS 84 / UTM zone 48S")
    ax.grid(True, color="0.85", lw=0.4, zorder=0)

    # --- legend: points + lines, single box, upper right (empty sea/land corner)
    handles = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", markeredgewidth=0.3, label=k)
               for k, c in cls_colors.items()]
    handles += [Line2D([], [], color=TAB10[0], lw=1.2, label="canal segment below MSL (z < 0)"),
                Line2D([], [], color="0.6", lw=1.0, label="canal / drain (OSM)"),
                Line2D([], [], color="k", lw=1.0, label="Kota Adm. Jakarta Utara")]
    ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=6.5,
              loc="center right", bbox_to_anchor=(0.995, 0.42), title="Stations / lines", title_fontsize=7)

    # --- north arrow (upper right) and 5 km scale bar (lower right, under legend)
    ax.annotate("N", xy=(0.96, 0.95), xytext=(0.96, 0.86), xycoords="axes fraction", ha="center", va="center",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2, shrinkA=0, shrinkB=0))
    ax.add_artist(AnchoredSizeBar(ax.transData, 5000, "5 km", "lower right", pad=0.5, sep=3,
                                  color="k", frameon=False, size_vertical=120, fontproperties=dict(size=7)))

    # --- CRS / datum / sources note, bottom-left inside the frame (nothing else lives there)
    ax.text(0.01, 0.01, "CRS: WGS 84 / UTM zone 48S (EPSG:32748), grid in metres\n"
            "Vertical: orthometric (DTM 1.5 m; geoid unconfirmed)\n"
            "Canal graph: OpenStreetMap contributors (ODbL) · boundary: Batas Kota DKI · stations: DSDA",
            transform=ax.transAxes, fontsize=5.5, color="0.3", va="bottom", zorder=6,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))

    # --- inset: western Java context (Natural Earth 10 m provinces), UTM metres, lower left
    ne = gpd.read_file(ROOT / "reference/naturalearth/ne10m_provinces_west_java.gpkg").to_crs(edges.crs)
    dki = gpd.read_file(ROOT / "reference/batas_adm/Batas Kota DKI.geojson").to_crs(edges.crs)
    ins = ax.inset_axes([0.02, 0.115, 0.30, 0.30])
    ins.set_facecolor("#dbe9f4")  # sea
    ne.plot(ax=ins, color="#f2efe9", edgecolor="0.55", lw=0.35)
    dki.dissolve().plot(ax=ins, color=TAB10[1], alpha=0.85, edgecolor="k", lw=0.4)
    ins.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                            fill=False, edgecolor="red", lw=0.8))
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    half = 220_000  # ~440 km window: Sunda Strait to Cirebon
    ins.set_xlim(cx - half, cx + half); ins.set_ylim(cy - half * 0.95, cy + half * 0.55)
    ins.set_aspect("equal")
    for nm, (dx, dy) in {"Jawa Barat": (60_000, -110_000), "Banten": (-110_000, -60_000),
                         "Lampung": (-175_000, 45_000)}.items():
        ins.text(cx + dx, cy + dy, nm, fontsize=4.5, color="0.35", ha="center", style="italic")
    ins.text(cx, cy + 22_000, "DKI Jakarta", fontsize=4.5, ha="center", va="bottom", fontweight="bold")
    ins.text(cx + 120_000, cy + 60_000, "Java Sea", fontsize=4.2, color="0.45", ha="center", style="italic")
    ins.xaxis.set_major_locator(MultipleLocator(200_000)); ins.yaxis.set_major_locator(MultipleLocator(200_000))
    ins.xaxis.set_major_formatter(m_fmt); ins.yaxis.set_major_formatter(m_fmt)
    ins.tick_params(labelsize=4.5, length=2, pad=1)
    for lab in ins.get_yticklabels():
        lab.set_rotation(90); lab.set_va("center")
    ins.text(0.5, 0.97, "Western Java — map extent in red", transform=ins.transAxes, ha="center", va="top",
             fontsize=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    for sp in ins.spines.values():
        sp.set_linewidth(0.5)
    save(fig, "fig4_7_canal_graph_stations")
    NOTES.append(("4.7", "UTM metre grid (northing labels rotated), inset of western Java (Natural Earth 10 m) with map extent, line+point legend, north arrow, 5 km scale bar, CRS/datum note (13 Sep). Station 150 Sunter Hulu lies upstream on Kali Sunter (Jakarta Timur), outside the Jakut boundary -- state in caption. Station 170 Ancol Flushing is NOT drawn (no published coordinates; NULL in `stations`) -- state in caption; "
                          "station 170 has no koordinat in stations.yaml, resolved via the "
                          "catalog `stations` table instead"))


README = """# Chapter 4 figures

| file | source data | draft note |
|---|---|---|
| fig4_1_station140_tma_flagged | `tinggi_air/data/processed/140_*_hourly.parquet` | 04-1 LSTM baseline and split decision |
| fig4_2_bclean_h6_obs_vs_pred | `reports/lstm_baseline/pred_B_clean_h6.parquet` | 04-1 LSTM baseline and split decision |
| fig4_3_nse_vs_horizon_configs | `reports/lstm_baseline/metrics.json` | 04-1 LSTM baseline and split decision |
| fig4_4_median_nse_vs_horizon_configs | `reports/compare_rain_loss/metrics.csv` | 04-2 Multi-station LSTM |
| fig4_5_event_E2_140_107 | `reports/lstm_multistation_gsmap_mse_persist_resid/pred_{140,107}_h6.parquet`, `forcing-acquisition/data/processed/forcing_hourly_multi_gsmap.parquet` | 04-2 Multi-station LSTM |
| fig4_6_f1_far_scatter | `reports/compare_rain_loss/metrics.csv` | 04-2 Multi-station LSTM |
| fig4_7_canal_graph_stations | `reference/jakarta/canal_graph.gpkg`, `reference/batas_adm/Batas Kota DKI.geojson`, `reference/naturalearth/ne10m_provinces_west_java.gpkg`, `catalog.duckdb` (`stations` table) / `tinggi_air/config/stations.yaml` | 04-2 Multi-station LSTM |

Each figure is exported as `.png` (200 dpi) and `.svg`. Generated by `scripts/fig_ch4.py`.
"""


def main():
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()
    fig_4_5()
    fig_4_6()
    fig_4_7()
    (OUT / "README.md").write_text(README)

    # self-check: 14 figure files + README, all under 2 MB
    fig_files = list(OUT.glob("fig4_*.png")) + list(OUT.glob("fig4_*.svg"))
    assert len(fig_files) == 14, f"expected 14 figure files, found {len(fig_files)}"
    for p in fig_files:
        assert p.stat().st_size < 2 * 1024 * 1024
    print(f"wrote {len(fig_files)} figure files + README.md to {OUT}")
    if NOTES:
        print("\napproximations:")
        for fig_id, note in NOTES:
            print(f"  [{fig_id}] {note}")


if __name__ == "__main__":
    main()
