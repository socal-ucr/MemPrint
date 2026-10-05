"""True footprint (full Pin trace) vs Valgrind massif peak heap per input size
(paper: <wl>_baseline_vs_massif.pdf)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import order_configs, save

UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_memory(text):
    """'22.34 KB' -> bytes."""
    value, unit = str(text).split()
    return float(value) * UNITS.get(unit, 1)


def plot_massif(workload, all_data, massif, out_dir):
    """massif: table with Size and PeakMemory columns (results/<wl>/massif.csv)."""
    truth = all_data[all_data["SamplingInterval"] == 1]
    massif = massif.copy()
    massif["PeakMemoryBytes"] = massif["PeakMemory"].apply(parse_memory)
    # Match configs as strings so numeric and text sizes line up.
    truth = truth.assign(key=truth["Config"].astype(str))
    massif = massif.assign(key=massif["Size"].astype(str))
    merged = truth.merge(massif, on="key")
    merged["Pct_Diff"] = (merged["PeakMemoryBytes"] - merged["MemUsageObs"]).abs() / merged["MemUsageObs"] * 100
    position = {c: i for i, c in enumerate(order_configs(merged["Config"]))}
    merged = merged.sort_values("Config", key=lambda s: s.map(position)).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(merged))
    ymax = float(np.nanmax(merged[["PeakMemoryBytes", "MemUsageObs"]].to_numpy()))
    ax.set_ylim(0, ymax * 1.1)

    for i, row in merged.iterrows():
        peak, true = row["PeakMemoryBytes"], row["MemUsageObs"]
        diff = abs(peak - true)
        ax.bar(i, peak, width=bar_width, color="C0")
        ax.bar(i + bar_width, true, width=bar_width, color="C2", alpha=0.5)
        # Hatched bar for the difference, stacked on the smaller of the two.
        if peak != true:
            x_low = i if peak < true else i + bar_width
            ax.bar(x_low, diff, width=bar_width * 0.95, bottom=min(peak, true), fill=False, edgecolor="black", hatch="//")
            ax.text(x_low, max(peak, true) * 1.01, f"{row['Pct_Diff']:.1f}%", ha="center", va="bottom", fontsize=12,
                    color="black", weight="bold")

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="C0", label="Massif Peak Memory (bytes)"),
            plt.Rectangle((0, 0), 1, 1, color="C2", label="True Memory Usage (bytes)"),
            plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", hatch="//",
                          label="Difference (percentage of True Usage)"),
        ],
        prop={"size": 14, "weight": "bold"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(merged["Config"])
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Input Size", fontsize=16, weight="bold")
    ax.set_ylabel("Memory Usage (bytes)", fontsize=16, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    save(plt, out_dir / f"{workload}_baseline_vs_massif.pdf")


def read_massif(path):
    return pd.read_csv(path, skipinitialspace=True)
