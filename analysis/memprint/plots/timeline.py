"""Footprint over time for the held-out configs: truth, reconstruction from
splitter bins and from a sampler run, and the forecast from half the run.
Sampler curves are drawn on the splitter run's time scale (fraction of the run)."""

import matplotlib.pyplot as plt

from ..timeline import truth_curves
from .common import save

MB = 1024 * 1024


def plot_timeline(workload, timeline, curves, out_dir, subset="L2O", variant="base", prefix=0.5):
    truths = {c: t for c, t in truth_curves(timeline).groupby("Config")}
    splits = list(dict.fromkeys(curves["split"]))
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        rows = curves[curves["split"] == split]
        config = rows["config"].iloc[0]
        truth = truths[config]
        ax.plot(truth["Time"], truth["Truth"] / MB, color="black", linewidth=2.5, label="True (splitter)")
        available = set(rows.loc[rows["source"] != "forecast", "subset"])
        use = subset if subset in available else "NZ"
        for source, style in [("splitter bins", dict(color="C0", linestyle="--")), ("sampler", dict(color="C1"))]:
            curve = rows[(rows["source"] == source) & (rows["subset"] == use) & (rows["variant"] == variant)]
            if len(curve):
                ax.plot(curve["Time"], curve["Estimate"] / MB, marker=".", markersize=3, linewidth=1.5,
                        label=f"Reconstructed from {source} ({use})", **style)
        forecast = rows[(rows["source"] == "forecast") & (rows["variant"] == f"prefix {prefix:g}")]
        if len(forecast):
            t0 = prefix * truth["Time"].max()
            ax.axvline(t0, color="gray", linestyle=":", linewidth=1)
            ax.plot(forecast["Time"], forecast["Estimate"] / MB, color="C2", linewidth=2,
                    label=f"Forecast from first {prefix:.0%}")
        ax.set_title(f"{workload} {config} ({'extrapolation' if split == 'EXTRA' else 'interpolation'})",
                     fontsize=13, weight="bold")
        ax.set_xlabel("Memory references executed", fontsize=12, weight="bold")
        ax.set_ylabel("Live footprint (MB)", fontsize=12, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out_dir / f"{workload}_timeline.pdf")
