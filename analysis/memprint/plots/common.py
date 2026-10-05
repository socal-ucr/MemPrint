"""Shared plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

# PolyBench dataset sizes, smallest first.
SIZE_ORDER = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM", "LARGE", "EXTRALARGE"]


def order_configs(configs):
    """Configs in increasing input size: numerically if they are numbers,
    otherwise by PolyBench size name (unknown names last, alphabetically)."""
    configs = list(dict.fromkeys(configs))
    numeric = pd.to_numeric(pd.Series(configs, dtype=object), errors="coerce")
    if numeric.notna().all():
        return [c for _, c in sorted(zip(numeric, configs))]
    rank = {name: i for i, name in enumerate(SIZE_ORDER)}
    return sorted(configs, key=lambda c: (rank.get(str(c), len(rank)), str(c)))


def bold_font(size):
    return FontProperties(size=size, weight="bold")


def save(fig_or_plt, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_or_plt.savefig(path, **kwargs)
    plt.close("all")
    print(f"wrote {path}")
