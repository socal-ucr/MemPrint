"""Spread of the observed footprint (sigma MemFP_obs) vs input size, one line per
sampling interval.

highlight:
  none  intervals 100..1000                       (paper: <wl>_sd_vs_config.pdf)
  best  intervals <= 10000, the monotone interval with the largest spread drawn
        solid and the others dashed               (paper: <wl>_sd_vs_config_highlight.pdf)
  mt    intervals <= 10000, every monotone interval drawn solid
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import NullFormatter

from ..dataset import prepare
from .common import bold_font, order_configs, save

SUFFIX = {"none": "", "best": "_highlight", "mt": "_highlight_MT"}


def is_monotone_increasing(y):
    return bool(np.all(np.diff(y) >= 0))


def spread_table(all_data, min_si, max_si):
    data = prepare(all_data).data
    data = data[(data["SamplingInterval"] >= min_si) & (data["SamplingInterval"] <= max_si)]
    table = data.groupby(["Config", "SamplingInterval"], as_index=False)["SD_MemUsage"].mean()
    configs = order_configs(table["Config"])
    table["pos"] = table["Config"].map({c: i for i, c in enumerate(configs)})
    return table.sort_values(["SamplingInterval", "pos"]), configs


def plot_sd_config(workload, all_data, out_dir, highlight="none"):
    min_si, max_si = (100, 1000) if highlight == "none" else (0, 10000)
    table, configs = spread_table(all_data, min_si, max_si)
    # Numeric configs (e.g. vertex counts) go on a log axis in the plain plot.
    log_x = highlight == "none" and pd.api.types.is_numeric_dtype(table["Config"])
    if log_x:
        table["pos"] = table["Config"]
    intervals = sorted(table["SamplingInterval"].unique())
    lines = {s: table[table["SamplingInterval"] == s] for s in intervals}
    palette = sns.color_palette("tab20", n_colors=len(intervals))
    color = {s: palette[i] for i, s in enumerate(intervals)}

    plt.figure(figsize=(9, 6))
    if highlight == "none":
        for s in intervals:
            plt.plot(lines[s]["pos"], lines[s]["SD_MemUsage"].values, marker="X", label=f"1 in {s}",
                     color=color[s], linewidth=5, markersize=8)
        legend_size = 14
    else:
        if highlight == "best":
            # Among monotone intervals, the one with the largest spread (then mean gap).
            def score(s):
                y = lines[s]["SD_MemUsage"].to_numpy()
                gap = float(np.mean(np.diff(y))) if len(y) > 1 else 0.0
                return (float(np.max(y) - np.min(y)), gap)

            monotone = [s for s in intervals if is_monotone_increasing(lines[s]["SD_MemUsage"].to_numpy())]
            highlighted = {max(monotone or intervals, key=score)}
            muted_style = dict(linewidth=3, alpha=0.9)
        else:
            highlighted = {
                s for s in intervals
                if len(lines[s]) == len(configs) and is_monotone_increasing(lines[s]["SD_MemUsage"].to_numpy())
            }
            muted_style = dict(linewidth=2.5, alpha=0.45)

        for s in intervals:
            if s in highlighted:
                continue
            plt.plot(lines[s]["pos"], lines[s]["SD_MemUsage"].to_numpy(), linestyle="--", marker="o", markersize=5,
                     color=sns.desaturate(color[s], 0.4), zorder=1, label=f"1 in {int(s)}", **muted_style)
        for s in intervals:
            if s not in highlighted:
                continue
            plt.plot(lines[s]["pos"], lines[s]["SD_MemUsage"].to_numpy(), linestyle="-", linewidth=5, marker="o",
                     markersize=8, color=color[s], zorder=3, label=f"★ 1 in {int(s)}")
        legend_size = 13

    handles, labels = plt.gca().get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: 0 if labels[i].startswith("★") else 1)
    plt.legend([handles[i] for i in order], [labels[i] for i in order], title="Sampling Rate",
               title_fontproperties=bold_font(16), bbox_to_anchor=(0.05, 1), loc="upper left",
               prop=bold_font(legend_size))

    plt.xlabel("Input Size", fontsize=16, weight="bold")
    plt.ylabel(r"$\sigma_{MemFP_{obs}}$", fontsize=18, weight="bold")
    plt.title(f"{workload}", fontsize=18, weight="bold")
    if log_x:
        plt.xscale("log")
    plt.xticks(ticks=configs if log_x else range(len(configs)), labels=[str(c) for c in configs], fontsize=14,
               weight="bold")
    plt.yticks(fontsize=12, weight="bold")
    if highlight == "none":
        plt.grid(True)
    else:
        plt.minorticks_off()
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    save(plt, out_dir / f"{workload}_sd_vs_config{SUFFIX[highlight]}.pdf")
