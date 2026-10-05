"""Alpha (true / observed footprint) for sampling intervals 1000..10000.

x="config": Alpha vs input size, one line per interval  (paper: <wl>_config_vs_alpha_title.pdf)
x="rate":   Alpha vs sampling rate, one line per input size (paper: <wl>_rate_vs_alpha_title.pdf)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from ..dataset import prepare
from .common import bold_font, order_configs, save


def plot_alpha(workload, all_data, out_dir, x="config"):
    data = prepare(all_data).data
    data = data[(data["SamplingInterval"] >= 1000) & (data["SamplingInterval"] <= 10000)].copy()
    configs = order_configs(data["Config"])
    numeric = pd.api.types.is_numeric_dtype(data["Config"])
    data["Input Size"] = data["Config"] if numeric else pd.Categorical(data["Config"], categories=configs, ordered=True)
    data = data.sort_values("Input Size", kind="stable").reset_index(drop=True)
    data["Sampling Rate"] = 1 / data["SamplingInterval"]

    if x == "config":
        x_col, hue, legend_title, xlabel = "Input Size", "SamplingInterval", "AvgSamplingInterval", "Input Size"
    else:
        x_col, hue, legend_title, xlabel = "Sampling Rate", "Input Size", "Input Size", "Sampling Rate (1 / AvgSamplingInterval)"

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=data, x=x_col, y="Alpha", hue=hue, marker="X", ms=10, palette="tab10", linewidth=5,
                 errorbar=None)
    plt.title(f"{workload}", size=18, weight="bold")
    plt.ylabel("Alpha", fontsize=16, weight="bold")
    plt.xlabel(xlabel, fontsize=16, weight="bold")
    if x == "rate":
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:.0e}"))
    plt.xticks(fontsize=14, weight="bold")
    plt.yticks(fontsize=14, weight="bold")
    plt.legend(title=legend_title, title_fontproperties=bold_font(16), loc="upper right",
               bbox_to_anchor=(0.98, 0.98), prop=bold_font(14))
    plt.grid(True)
    plt.tight_layout()
    name = "config_vs_alpha" if x == "config" else "rate_vs_alpha"
    save(plt, out_dir / f"{workload}_{name}_title.pdf")
