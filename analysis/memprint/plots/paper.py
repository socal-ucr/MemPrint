"""Paper figures whose data was measured by hand (runtime overheads and the
memory comparison against Valgrind). The measurements are kept here, once."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .common import save

VERTICES = [1024, 2048, 4096, 8192, 16384]

# miniVite under the sampler: (vertices, native s, raw -i, Pin s, Pin overhead s).
# -i 1000000000000 samples nothing (instrumentation only).
INSTR_ONLY = 1000000000000
SAMPLER_RUNS = pd.DataFrame([
    (1024, .064029, 10, 82.364579, 82.300550), (1024, .056131, 25, 29.214732, 29.158601),
    (1024, .060133, 50, 25.290272, 25.230139), (1024, .057399, 75, 24.344748, 24.287349),
    (1024, .059307, 100, 24.023099, 23.963792), (1024, .057963, 250, 23.034269, 22.976306),
    (1024, .056492, 500, 22.891600, 22.835108), (1024, .058057, 1000, 23.015472, 22.957415),
    (1024, .056793, 2500, 22.809179, 22.752386), (1024, .058632, 5000, 22.715904, 22.657272),
    (1024, .057742, 7500, 22.740476, 22.682734), (1024, .060275, 10000, 22.882859, 22.822584),
    (1024, .058455, INSTR_ONLY, 22.734511, 22.676056),
    (2048, .071729, 10, 126.497180, 126.425451), (2048, .064802, 25, 32.367097, 32.302295),
    (2048, .063585, 50, 26.499901, 26.436316), (2048, .063870, 75, 25.403255, 25.339385),
    (2048, .063982, 100, 24.708924, 24.644942), (2048, .064904, 250, 23.668861, 23.603957),
    (2048, .067456, 500, 23.163250, 23.095794), (2048, .064870, 1000, 23.364375, 23.299505),
    (2048, .067434, 2500, 23.099935, 23.032501), (2048, .067855, 5000, 23.020088, 22.952233),
    (2048, .065221, 7500, 22.933863, 22.868642), (2048, .065169, 10000, 23.014128, 22.948959),
    (2048, .067036, INSTR_ONLY, 22.938595, 22.871559),
    (4096, .093401, 10, 119.342553, 119.249152), (4096, .088613, 25, 38.408782, 38.320169),
    (4096, .085951, 50, 29.584588, 29.498637), (4096, .084984, 75, 27.755149, 27.670165),
    (4096, .085312, 100, 26.478455, 26.393143), (4096, .084756, 250, 24.728514, 24.643758),
    (4096, .085608, 500, 24.207518, 24.121910), (4096, .086722, 1000, 23.938756, 23.852034),
    (4096, .086906, 2500, 23.605315, 23.518409), (4096, .085848, 5000, 23.554099, 23.468251),
    (4096, .090622, 7500, 23.444094, 23.353472), (4096, .084753, 10000, 23.505697, 23.420944),
    (4096, .091115, INSTR_ONLY, 23.332441, 23.241326),
    (8192, .178881, 10, 185.409034, 185.230153), (8192, .176144, 25, 64.829302, 64.653158),
    (8192, .171861, 50, 42.289091, 42.117230), (8192, .170197, 75, 35.981717, 35.811520),
    (8192, .172266, 100, 33.266907, 33.094641), (8192, .170579, 250, 29.100966, 28.930387),
    (8192, .169371, 500, 27.322431, 27.153060), (8192, .173420, 1000, 26.563392, 26.389972),
    (8192, .172907, 2500, 26.043531, 25.870624), (8192, .169796, 5000, 25.648217, 25.478421),
    (8192, .169357, 7500, 25.651548, 25.482191), (8192, .176460, 10000, 25.724577, 25.548117),
    (8192, .173888, INSTR_ONLY, 25.478931, 25.305043),
    (16384, .491953, 10, 298.583003, 298.091050), (16384, .487190, 25, 129.773356, 129.286166),
    (16384, .488573, 50, 76.023571, 75.534998), (16384, .485655, 75, 64.044747, 63.559092),
    (16384, .482702, 100, 53.431503, 52.948801), (16384, .484222, 250, 40.868787, 40.384565),
    (16384, .488210, 500, 36.618710, 36.130500), (16384, .487601, 1000, 34.709622, 34.222021),
    (16384, .488107, 2500, 33.027813, 32.539706), (16384, .494139, 5000, 32.593220, 32.099081),
    (16384, .483179, 7500, 32.488622, 32.005443), (16384, .485838, 10000, 32.215358, 31.729520),
    (16384, .482788, INSTR_ONLY, 31.631165, 31.148377),
], columns=["vertices", "avg_runtime_without_pin", "SI", "avg_w_PIN", "PIN_overhead"])

# The sampler ran with -s 10 -r 1, so each bin's effective interval is 10x -i.
SAMPLER_BIN_FACTOR = 10

# miniVite under the splitter (full trace): total and instrumentation-only overhead (s).
SPLITTER_TOTAL = [216.375, 267.397, 235.964, 373.382, 557.326]
SPLITTER_INSTR = [24.149, 25.277, 27.150, 36.004, 57.811]

# Sampler overhead breakdown for three effective intervals (s).
SAMPLER_INSTR = [22.676, 22.872, 23.241, 25.305, 31.148]
SAMPLER_ANALYSIS = {
    500: [2.554, 3.564, 6.258, 16.812, 44.387],
    25000: [0.076, 0.161, 0.277, 0.566, 1.392],
    50000: [0.019, 0.080, 0.227, 0.173, 0.951],
}

# miniVite test MAPE (%) when training on inputs up to n vertices.
CUMULATIVE_TRAINING = ["1024", "2048", "4096", "8192"]
CUMULATIVE_MAPE = {
    "NZ": [26.93, 14.45, 15.07, 3.98],
    "MT": [78.49, 15.35, 25.83, 4.98],
    "L2O": [26.93, 16.53, 5.24, 12.90],
}

# miniVite peak memory (bytes; Massif in MB) from each tool.
MEMORY_COMPARISON = pd.DataFrame({
    "Size": VERTICES,
    "Massif": ["5.245 MB", "5.245 MB", "5.317 MB", "6.699 MB", "9.470 MB"],
    "DHAT": [5489677, 5489677, 5489677, 6685195, 9591502],
    "PIN": [6843827, 7323781, 8266151, 10206140, 13969068],
    "MemPrint-NZ": [6800771, 7347398, 8023224, 9694065, 13413583],
    "MemPrint-L2O": [6304041, 7683711, 8135997, 10582894, 12167082],
    "MemPrint-MT": [7082213, 7717698, 8180011, 10448654, 13273988],
})


def lighten(color, amount=0.6):
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - (1 - c) * (1 - amount))


def pin_overhead(out_dir):
    """Sampler overhead vs effective sampling interval (miniVite_PIN_overhead.pdf)."""
    runs = SAMPLER_RUNS[SAMPLER_RUNS["SI"] != INSTR_ONLY].copy()
    runs["SI_scaled"] = runs["SI"] * SAMPLER_BIN_FACTOR
    viridis = ["#440154", "#31688e", "#35b779", "#fde725", "#3e4989"]
    color = {size: viridis[i % len(viridis)] for i, size in enumerate(sorted(runs["vertices"].unique()))}

    plt.figure(figsize=(10, 6))
    for size, group in runs.groupby("vertices"):
        plt.plot(group["SI_scaled"], group["PIN_overhead"], marker="o", linewidth=5, markersize=8,
                 label=f"Vertices={size}", color=color[size])
    plt.xscale("log")
    plt.xlabel("Average Sampling Interval (log scale)", fontsize=16, fontweight="bold", labelpad=6)
    plt.ylabel("Total PIN Overhead (s)", fontsize=16, fontweight="bold", labelpad=6)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.ylim(bottom=0)
    plt.legend(loc="best", frameon=True, prop={"size": 14, "weight": "bold"})
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    save(plt, out_dir / "miniVite_PIN_overhead.pdf")


def splitter_overhead(out_dir):
    """Instrumentation + analysis overhead of the splitter (pin_splitter_overhead.pdf)."""
    analysis = np.array(SPLITTER_TOTAL) - np.array(SPLITTER_INSTR)
    x = np.arange(len(VERTICES))
    width = 0.5
    base = "#0072B2"

    plt.figure(figsize=(6, 4))
    plt.bar(x, SPLITTER_INSTR, width, color=base, edgecolor="black", linewidth=0.4, label="Instrumentation")
    plt.bar(x, analysis, width, bottom=SPLITTER_INSTR, color=lighten(base, 0.6), edgecolor="black", linewidth=0.4,
            label="Analysis")
    for xi, yi in zip(x, SPLITTER_TOTAL):
        plt.text(xi, yi + 10, f"{yi:.1f}", ha="center", fontsize=10)
    for xi, yi in zip(x, SPLITTER_INSTR):
        plt.text(xi - width / 2 - 0.05, yi, f"{yi:.1f}", ha="right", va="center", fontsize=10, weight="bold")
    plt.xticks(x, VERTICES, weight="bold")
    plt.xlabel("Vertices", weight="bold")
    plt.ylabel("PIN SPLITTER Overhead (s)", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xlim(-0.7, len(x) - 1 + 0.5)
    plt.ylim(0, max(SPLITTER_TOTAL) + 50)
    plt.legend(fontsize=12)
    plt.tight_layout()
    save(plt, out_dir / "pin_splitter_overhead.pdf", bbox_inches="tight")


def sampler_overhead(out_dir):
    """Instrumentation + analysis overhead of the sampler (pin_sampler_overhead.pdf)."""
    colors = {500: "#0072B2", 25000: "#E69F00", 50000: "#009E73"}
    x = np.arange(len(VERTICES))
    width = 0.22

    plt.figure(figsize=(7, 4))
    for j, interval in enumerate(SAMPLER_ANALYSIS):
        xpos = x + (j - 1) * width
        plt.bar(xpos, SAMPLER_INSTR, width, color=colors[interval], edgecolor="black", linewidth=0.3)
        plt.bar(xpos, SAMPLER_ANALYSIS[interval], width, bottom=SAMPLER_INSTR, color=lighten(colors[interval], 0.6),
                edgecolor="black", linewidth=0.3)
        total = np.array(SAMPLER_INSTR) + np.array(SAMPLER_ANALYSIS[interval])
        for xi, yi in zip(xpos, total):
            plt.text(xi, yi + 1, f"{yi:.1f}", ha="center", fontsize=7, weight="bold")
    plt.xticks(x, VERTICES, weight="bold")
    plt.xlabel("Vertices", weight="bold")
    plt.ylabel("PIN SAMPLER Overhead (s)", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, yi in zip(x, SAMPLER_INSTR):
        plt.text(xi - width - 0.13, yi, f"{yi:.1f}", ha="right", va="center", fontsize=8, weight="bold")
    plt.legend(
        handles=[Patch(facecolor=colors[i], label=f"SI={i}") for i in SAMPLER_ANALYSIS]
        + [Patch(facecolor="gray", alpha=0.9, label="Instrumentation"),
           Patch(facecolor="gray", alpha=0.3, label="Analysis")],
        fontsize=12,
        ncol=2,
    )
    plt.xlim(-0.7, len(x) - 1 + 0.5)
    plt.tight_layout()
    save(plt, out_dir / "pin_sampler_overhead.pdf", bbox_inches="tight")


def cumulative_accuracy(out_dir):
    """miniVite test MAPE vs largest training input (cumulative_overhead_vs_accuracy.pdf)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, values), marker in zip(CUMULATIVE_MAPE.items(), ["o", "s", "^"]):
        ax.plot(CUMULATIVE_TRAINING, values, marker=marker, markersize=8, linewidth=3, label=label)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylabel("MAPE (%)", fontsize=18, weight="bold")
    plt.subplots_adjust(left=0.16, bottom=0.23)
    ax.tick_params(axis="x", labelsize=16, rotation=0)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_xlabel("Training data upto n vertices", fontsize=18, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", prop={"size": 18, "weight": "bold"})
    plt.tight_layout()
    save(plt, out_dir / "cumulative_overhead_vs_accuracy.pdf", format="pdf")


def memory_comparison(out_dir):
    """Peak memory from Pin, Massif, DHAT and MemPrint for miniVite (memory_usage_comparison.pdf)."""
    df = MEMORY_COMPARISON.copy()
    df["Massif"] = df["Massif"].apply(lambda mb: float(mb.replace(" MB", "")) * 1024 * 1024).astype(int)
    methods = ["Massif", "DHAT", "MemPrint-NZ", "MemPrint-MT", "MemPrint-L2O"]
    labels = ["Valgrind-Massif", "Valgrind-DHAT", "MemPrint-NZ", "MemPrint-MT", "MemPrint-L2O"]
    colors = ["#99ccff", "#3366cc", "#ffcc99", "#ff9966", "#ff6600"]
    bar_width = 0.15
    offsets = [-bar_width, 0, bar_width, 2 * bar_width, 3 * bar_width]
    x = np.arange(len(df))

    plt.figure(figsize=(12, 7))
    plt.bar(x - 2 * bar_width, df["PIN"], width=bar_width, label="PIN (True)", color="black")
    for method, label, offset, color in zip(methods, labels, offsets, colors):
        plt.bar(x + offset, df[method], width=bar_width, label=label, color=color)
    for method, offset, color in zip(methods, offsets, colors):
        errors = (df[method] - df["PIN"]) / df["PIN"] * 100
        for i, value in enumerate(df[method]):
            plt.text(i + offset, value + 150000, f"{errors.iloc[i]:.1f}%", ha="center", va="bottom", fontsize=18,
                     rotation=90, color=color, fontweight="bold")
    plt.xticks(x, df["Size"], fontsize=18, fontweight="bold")
    plt.xlabel("Input Size", fontsize=22, fontweight="bold")
    plt.ylabel("Memory Usage (Bytes)", fontsize=22, fontweight="bold")
    plt.legend(fontsize=18)
    plt.ylim(0, max(df["PIN"].max(), df["MemPrint-MT"].max()) * 1.25)
    plt.tight_layout()
    save(plt, out_dir / "memory_usage_comparison.pdf", format="pdf")


FIGURES = {
    "pin-overhead": pin_overhead,
    "splitter-overhead": splitter_overhead,
    "sampler-overhead": sampler_overhead,
    "cumulative-accuracy": cumulative_accuracy,
    "memory-comparison": memory_comparison,
}
