"""Brief §7: plot hujan-vs-TMA untuk kejadian banjir yang diketahui. Kalau
puncak hujan tidak mendahului puncak TMA dengan lag masuk akal, ada yang
salah di alignment temporal - berhenti sebelum lanjut ke pemodelan.

Kejadian yang dipakai: banjir besar Jakarta 19-20 Feb 2021 (dipilih karena
well-documented DAN berada di dalam rentang data TMA node 140, yang mulai
2021-01-01 - banjir 1 Jan 2020 di luar rentang ini, tidak dipakai). Event
lain di luar Feb 2021 tidak ditambahkan tanpa verifikasi tanggal yang pasti
- lebih baik satu event yang benar daripada beberapa yang ditebak.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"

EVENTS = {
    "banjir_19-20_feb_2021": ("2021-02-15", "2021-02-25"),
}


def plot_event(df: pd.DataFrame, name: str, start: str, end: str) -> Path:
    window = df[(df["waktu"] >= start) & (df["waktu"] <= end)]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(window["waktu"], window["hujan_mm"] / 24, width=0.03, color="tab:blue", alpha=0.5,
            label="hujan (mm/hari, ditampilkan per-jam)")
    ax1.set_ylabel("hujan (mm/24h, digambar per jam)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(window["waktu"], window["tma_max"], color="tab:red", label="TMA max (cm)")
    ax2.set_ylabel("TMA (cm)", color="tab:red")
    fig.suptitle(f"Sanity check: hujan vs TMA - {name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sanity_check_{name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "processed" / "forcing_hourly.parquet")
    df["waktu"] = pd.to_datetime(df["waktu"])
    for name, (start, end) in EVENTS.items():
        p = plot_event(df, name, start, end)
        print("saved", p)
