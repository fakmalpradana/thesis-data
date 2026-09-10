"""LSTM baseline results — TMA forecasting at station 140, split decision.

    marimo edit notebooks/02_lstm_baseline.py

Reads reports/lstm_baseline/metrics.json + pred_*.parquet (produced by
experiments/lstm_baseline.py). Overview -> interactive view, same pattern as
explore.py.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    REPORT_DIR = ROOT / "reports" / "lstm_baseline"
    metrics = pd.DataFrame(json.loads((REPORT_DIR / "metrics.json").read_text()))
    return REPORT_DIR, metrics, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # LSTM baseline — station 140, split decision

    An LSTM (1 layer, hidden 64) forecasting `tma_mean` at horizons {1, 6, 12, 24} h from
    72 h of history (`tma_mean`, `tide_m`, `hujan_mm`) plus the astronomical `tide_m` at the
    target hour (known in advance, not leakage). Three train/val/test splits run side by
    side to answer the pending split question: does the post-2024 sensor degradation
    (see `explore.py` Sec. 2) force a pre-2024 split, or does a clean-flag filter fix it?

    - **A_chrono** — train 2021-2023, val 2024, test 2025-2026, all QC flags kept.
    - **B_clean** — same ranges, but any window touching `tma_qc_flag != 0` is dropped.
    - **C_pre2024** — train 2021-01→2022-06, val 2022-07→2022-12, test 2023, all flags.

    Persistence (predict `tma_mean[t]` at `t+h`) is the baseline every config must beat.
    Metrics from `experiments/lstm_baseline.py`, run separately (`python3 experiments/lstm_baseline.py`).
    """)
    return


@app.cell
def _(metrics, mo):
    _piv = metrics.pivot(index="horizon", columns="config", values=["lstm_nse", "persistence_nse"]).round(4)
    _piv.columns = [f"{m}_{c}" for m, c in _piv.columns]
    _piv = _piv.reset_index()
    mo.vstack(
        [
            mo.md("## 1. Metrics — config x horizon, LSTM NSE vs persistence NSE"),
            mo.ui.table(metrics.round(4), selection=None, page_size=15),
            mo.md("Pivoted view (NSE only):"),
            mo.ui.table(_piv, selection=None),
        ]
    )
    return


@app.cell
def _(metrics, mo):
    configs = sorted(metrics["config"].unique())
    horizons = sorted(metrics["horizon"].unique())
    pick_config = mo.ui.dropdown(configs, value=configs[0], label="config")
    pick_horizon = mo.ui.dropdown([str(h) for h in horizons], value=str(horizons[0]), label="horizon (h)")
    mo.vstack([mo.md("## 2. Predictions — y vs yhat"), mo.hstack([pick_config, pick_horizon])])
    return pick_config, pick_horizon


@app.cell
def _(REPORT_DIR, mo, pd, pick_config, pick_horizon, plt):
    _pred = pd.read_parquet(REPORT_DIR / f"pred_{pick_config.value}_h{pick_horizon.value}.parquet")
    _fig, _ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    _ax[0].plot(_pred.waktu, _pred.y, lw=0.4, label="observed")
    _ax[0].plot(_pred.waktu, _pred.yhat, lw=0.4, alpha=0.7, label="LSTM")
    _ax[0].set_ylabel("tma_mean (cm)")
    _ax[0].legend()
    _ax[1].scatter(_pred.y, _pred.yhat, s=2, alpha=0.3, c=(_pred.qc_flag != 0).map({True: "tab:red", False: "tab:blue"}))
    _lim = [_pred.y.min(), _pred.y.max()]
    _ax[1].plot(_lim, _lim, "k--", lw=0.8)
    _ax[1].set_xlabel("observed")
    _ax[1].set_ylabel("predicted")
    _fig.suptitle(f"{pick_config.value}, h={pick_horizon.value} (red = flagged)")
    mo.vstack([_fig, mo.ui.table(_pred, page_size=10)])
    return


@app.cell
def _(metrics, mo):
    _a = metrics[metrics.config == "A_chrono"].set_index("horizon")
    _b = metrics[metrics.config == "B_clean"].set_index("horizon")
    _c = metrics[metrics.config == "C_pre2024"].set_index("horizon")
    _a_beats = (_a.lstm_nse > _a.persistence_nse).all()
    _b_gain = (_b.lstm_nse - _a.lstm_nse).mean()
    _c_gain = (_c.lstm_nse - _a.lstm_nse).mean()
    _best = "B_clean" if _b_gain > max(_c_gain, 0) else ("C_pre2024" if _c_gain > 0 else "A_chrono")
    mo.md(
        f"""
        ## 3. Split decision

        LSTM beats persistence on every horizon under A_chrono: **{_a_beats}**. Mean LSTM NSE
        gain of B_clean over A_chrono (same date ranges, flagged windows dropped):
        **{_b_gain:+.4f}**. Mean gain of C_pre2024 (pre-degradation window only) over A_chrono:
        **{_c_gain:+.4f}**.

        **Recommendation: use `{_best}`.** {"Filtering flagged windows (B_clean) recovers "
        "most of the post-2024 degradation without shrinking the usable time range as much as "
        "restricting to pre-2024 data — keep the full chronology, drop flagged windows at train "
        "and eval time." if _best == "B_clean" else
        "The post-2024 degradation is severe enough that even flag-filtering doesn't match "
        "training only on the clean pre-2024 period." if _best == "C_pre2024" else
        "Flag-filtering and pre-2024 restriction both underperform the full chronological "
        "split — the extra 2024-2026 training data outweighs its noisier flag share."}
        """
    )
    return


if __name__ == "__main__":
    app.run()
