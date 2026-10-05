"""MemPrint analysis pipeline.

    python -m memprint [--root DIR] COMMAND ...

Inputs and outputs live under --root (default: current directory):
    traces/<workload>/     splitter traces from scripts/run.sh
    results/<workload>/    massif.csv from scripts/run.sh --massif
    data/                  tables written by these commands
    figures/               plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import build, predict, timeline, traces, transform
from .model import COEF_COLS

WORKLOAD_LIST = Path(__file__).resolve().parent.parent / "workloads.txt"

# Workloads left out of the cross-workload figures (outliers that dominate the colour scale).
HEATMAP_EXCLUDE = ["bicg"]
ERROR_MATRIX_EXCLUDE = ["miniVite", "bicg", "gesummv", "mvt", "trisolv", "atax", "gemver", "doitgen"]
TRANSFORM_EXCLUDE = ["bicg", "gesummv", "mvt", "trisolv", "gemver"]

# Paper figure -> how to make it (see README).
PAPER_WORKLOADS = {
    "massif": ["gemm", "miniVite"],
    "sd-config": ["gemm", "miniVite"],
    "alpha": ["gemm", "miniVite"],
    "sd-config-best": ["gemver", "gemm", "floyd-warshall", "miniVite"],
}


class Paths:
    def __init__(self, root):
        self.root = Path(root)
        self.traces = self.root / "traces"
        self.results = self.root / "results"
        self.data = self.root / "data"
        self.figures = self.root / "figures"

    def all_data(self, workload):
        return self.data / f"{workload}_allData.csv"

    def read_all_data(self, workload):
        return pd.read_csv(self.all_data(workload))

    @property
    def models(self):
        return self.data / "models.csv"

    def errors(self, split):
        return self.data / f"{'extrapolation' if split == 'EXTRA' else 'interpolation'}_errors.csv"

    def timeline(self, workload):
        return self.data / f"{workload}_timeline.csv"

    def read_timeline(self, workload):
        return pd.read_csv(self.timeline(workload), dtype={"Config": str, "PID": str})

    def split_models(self, split):
        return self.data / f"{'extrapolation' if split == 'EXTRA' else 'interpolation'}_models.csv"


def default_workloads():
    return WORKLOAD_LIST.read_text().split()


def write(frame, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, **kwargs)
    print(f"wrote {path}")


# ---------------------------------------------------------------- commands


def cmd_preprocess(args, paths):
    for workload in args.workloads or default_workloads():
        table = traces.add_sample_spread(traces.load_traces(paths.traces / workload, prefix=args.prefix))
        write(table, paths.all_data(workload), index=False)


def cmd_build(args, paths):
    fits = []
    for workload in args.workloads or default_workloads():
        print(f"fitting {workload}")
        fits += build.fit_workload(workload, paths.read_all_data(workload))
    write(build.select_models(fits), paths.models, index=False)
    write(build.fits_table(fits), paths.data / "fits.csv", index=False)
    if args.latex:
        for name, lines in build.latex_tables(fits).items():
            print(f"\n% {name}")
            print("\n".join(lines))


def cmd_predict(args, paths):
    models = pd.read_csv(paths.models)
    if args.coef_decimals is not None:
        # The original pipeline stored coefficients with 4 decimals; rounding
        # reproduces its cross-workload errors exactly.
        for col in COEF_COLS:
            models[col] = models[col].map(lambda v: float(f"{v:.{args.coef_decimals}f}"))
    workloads = list(dict.fromkeys(models["workload"]))
    all_data = {w: paths.read_all_data(w) for w in workloads}
    for split in build.SPLITS:
        write(predict.error_matrix(models, all_data, split), paths.errors(split))
        write(predict.model_table(models, split), paths.split_models(split), index=False)


def cmd_transform(args, paths):
    from .plots.heatmaps import plot_transformation

    models = pd.read_csv(paths.split_models("EXTRA"))
    result = transform.transform(models, predict.read_matrix(paths.errors("EXTRA")))
    print(f"RMSE: {result.rmse}\nMAPE: {result.mape}")
    write(result.transformation, paths.data / "transformation_matrix.csv")
    write(result.similarity, paths.data / "model_similarity_rbf.csv")
    plot_transformation(result.transformation, paths.figures, exclude=args.exclude)


def cmd_timeline(args, paths):
    workloads = args.workloads or default_workloads()
    if args.action == "preprocess":
        for w in workloads:
            write(traces.load_timelines(paths.traces / w), paths.timeline(w), index=False)
        return

    if args.action == "build":
        recon, fcast, curves, models = [], [], [], []
        for w in workloads:
            print(f"evaluating {w}")
            tl = paths.read_timeline(w)
            r, f, c = timeline.evaluate(w, tl)
            recon.append(r)
            fcast.append(f)
            curves.append(c)
            model, si, subset = timeline.fit_final_model(tl, subset=args.subset, variant=args.variant)
            models.append([w, subset, args.variant, si, model.intercept] + list(model.coef))
        recon, fcast = pd.concat(recon), pd.concat(fcast)
        write(recon, paths.data / "timeline_reconstruction.csv", index=False)
        write(fcast, paths.data / "timeline_forecast.csv", index=False)
        write(pd.concat(curves), paths.data / "timeline_curves.csv", index=False)
        extra = [f"b{i}" for i in range(8, 8 + len(timeline.VARIANTS[args.variant]))]
        write(pd.DataFrame(models, columns=["workload", "subset", "variant", "sample_rate"] + COEF_COLS + extra),
              paths.data / "timeline_models.csv", index=False)
        summarize_timeline(recon, fcast)
        return

    # estimate / forecast a single sampler run
    models = pd.read_csv(paths.data / "timeline_models.csv").set_index("workload")
    for w in workloads:
        row = models.loc[w]
        coef_cols = [c for c in models.columns if c.startswith("b")]
        model = timeline.Model(float(row["intercept"]), row[coef_cols].dropna().to_numpy(float),
                               timeline.VARIANTS[row["variant"]])
        runs = traces.load_timelines(Path(args.run) if args.run else paths.traces / w, prefixes=("Sampled",))
        bins = timeline.bin_rows(runs, "sampler")
        for (config, pid), run in bins.groupby(["Config", "PID"]):
            si = timeline.closest(sorted(run["SamplingInterval"].unique()), row["sample_rate"])
            curve = timeline.estimate_curve(run[run["SamplingInterval"] == si], model)
            if args.action == "forecast":
                prefix = curve[curve["Time"] <= args.upto].rename(columns={"Estimate": "Truth"})
                templates = [t for _, t in timeline.truth_curves(paths.read_timeline(w)).groupby("Config")]
                fc = timeline.forecast(prefix, templates)
                curve = pd.concat([curve.assign(Kind="estimate"),
                                   fc.curve.rename(columns={"Forecast": "Estimate"}).assign(Kind="forecast")])
                print(f"{w} {config}: forecast end at {fc.end_time:.4g} references, final {fc.final:.4g} B, "
                      f"peak {fc.peak:.4g} B")
            write(curve, paths.data / f"{w}-{config}_{pid}_{args.action}.csv", index=False)


def summarize_timeline(recon, fcast):
    pd.set_option("display.width", 200)
    print("\nReconstruction MAPE (%) over snapshots, held-out config, interval with min training MAPE:")
    print(recon.pivot_table(index=["workload", "split"], columns=["subset", "variant"],
                            values=["splitter_mape", "sampler_mape"], aggfunc="first").round(2).to_string())
    print("\nMean absolute value over workloads:")
    print(recon.groupby(["split", "subset", "variant"])[["splitter_mape", "splitter_peak_error", "sampler_mape",
                                                         "sampler_peak_error", "sampler_length_error"]]
          .agg(lambda x: np.mean(np.abs(x))).round(2).to_string())
    print("\nForecast from the true prefix (mean absolute % error over workloads):")
    print(fcast.groupby(["split", "prefix"])[["rest_mape", "peak_error", "end_error"]]
          .agg(lambda x: np.mean(np.abs(x))).round(2).to_string())


def cmd_plot(args, paths):
    from .plots import alpha, heatmaps, massif, paper, sd_config

    out = Path(args.out) if args.out else paths.figures
    if args.figure == "sd-config":
        for w in args.workloads:
            sd_config.plot_sd_config(w, paths.read_all_data(w), out, highlight=args.highlight)
    elif args.figure == "alpha":
        for w in args.workloads:
            alpha.plot_alpha(w, paths.read_all_data(w), out, x=args.x)
    elif args.figure == "massif":
        for w in args.workloads:
            csv = Path(args.massif_csv.format(workload=w)) if args.massif_csv else paths.results / w / "massif.csv"
            massif.plot_massif(w, paths.read_all_data(w), massif.read_massif(csv), out)
    elif args.figure == "heatmaps":
        for split in build.SPLITS:
            heatmaps.plot_model_distances(pd.read_csv(paths.split_models(split)), split, out,
                                          exclude=args.exclude or HEATMAP_EXCLUDE)
        for split, name in [("EXTRA", "extrapolate"), ("INTER", "interpolate")]:
            heatmaps.plot_error_matrix(predict.read_matrix(paths.errors(split)), name, out,
                                       exclude=args.exclude or ERROR_MATRIX_EXCLUDE)
    elif args.figure == "timeline":
        from .plots import timeline as timeline_plot

        curves = pd.read_csv(paths.data / "timeline_curves.csv", dtype={"config": str})
        for w in args.workloads:
            timeline_plot.plot_timeline(w, paths.read_timeline(w), curves[curves["workload"] == w], out)
    elif args.figure == "paper":
        for name in args.workloads or list(paper.FIGURES):
            paper.FIGURES[name](out)


def cmd_paper_figures(args, paths):
    """Every data figure of the paper, under its file name in the paper."""
    from .plots import alpha, massif, paper, sd_config

    out = Path(args.out) if args.out else paths.figures / "paper"
    for w in PAPER_WORKLOADS["massif"]:
        csv = Path(args.massif_csv.format(workload=w)) if args.massif_csv else paths.results / w / "massif.csv"
        massif.plot_massif(w, paths.read_all_data(w), massif.read_massif(csv), out)
    for w in PAPER_WORKLOADS["sd-config"]:
        sd_config.plot_sd_config(w, paths.read_all_data(w), out, highlight="none")
    for w in PAPER_WORKLOADS["alpha"]:
        alpha.plot_alpha(w, paths.read_all_data(w), out, x="config")
        alpha.plot_alpha(w, paths.read_all_data(w), out, x="rate")
    for w in PAPER_WORKLOADS["sd-config-best"]:
        sd_config.plot_sd_config(w, paths.read_all_data(w), out, highlight="best")
    for name in ["pin-overhead", "splitter-overhead", "cumulative-accuracy", "memory-comparison"]:
        paper.FIGURES[name](out)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="memprint", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="directory holding traces/, results/, data/, figures/")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="traces/<wl>/*.csv -> data/<wl>_allData.csv")
    p.add_argument("workloads", nargs="*", help=f"default: the workloads in {WORKLOAD_LIST.name}")
    p.add_argument("--prefix", default="Buffered", help="trace file prefix (Buffered = splitter traces)")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("build", help="fit models -> data/models.csv, data/fits.csv")
    p.add_argument("workloads", nargs="*")
    p.add_argument("--latex", action="store_true", help="print the rows of the paper's accuracy tables")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("predict", help="cross-workload error matrices -> data/*_errors.csv")
    p.add_argument("--coef-decimals", type=int,
                   help="round model coefficients first (4 reproduces the published cross-workload errors)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("transform", help="model-similarity transformation of the extrapolation errors")
    p.add_argument("--exclude", nargs="*", default=TRANSFORM_EXCLUDE, help="workloads left out of the heatmap")
    p.set_defaults(func=cmd_transform)

    p = sub.add_parser("plot", help="one kind of figure")
    p.add_argument("figure", choices=["sd-config", "alpha", "massif", "heatmaps", "timeline", "paper"])
    p.add_argument("workloads", nargs="*", help="workloads (for paper: figure names, default all)")
    p.add_argument("--highlight", choices=["none", "best", "mt"], default="none", help="sd-config")
    p.add_argument("--x", choices=["config", "rate"], default="config", help="alpha")
    p.add_argument("--massif-csv", help="massif: CSV path, may contain {workload} (default results/<wl>/massif.csv)")
    p.add_argument("--exclude", nargs="*", help="heatmaps: workloads to leave out")
    p.add_argument("--out", help="output directory (default figures/)")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("timeline", help="footprint over time (-snapshot traces)")
    p.add_argument("action", choices=["preprocess", "build", "estimate", "forecast"],
                   help="preprocess: traces/<wl>/*_timeline.csv -> data/<wl>_timeline.csv; "
                        "build: evaluate and fit models -> data/timeline_*.csv; "
                        "estimate/forecast: apply the model to sampler runs")
    p.add_argument("workloads", nargs="*")
    p.add_argument("--subset", choices=["NZ", "MT", "L2O"], default="L2O", help="build: training subset of the final model")
    p.add_argument("--variant", choices=list(timeline.VARIANTS), default="base",
                   help="build: model features (time adds log Time)")
    p.add_argument("--run", help="estimate/forecast: directory with the sampler timelines (default traces/<wl>)")
    p.add_argument("--upto", type=float, help="forecast: use the run up to this many memory references")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("paper-figures", help="regenerate the paper's data figures into figures/paper/")
    p.add_argument("--massif-csv", help="CSV path, may contain {workload} (default results/<wl>/massif.csv)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_paper_figures)

    args = parser.parse_args(argv)
    args.func(args, Paths(args.root))


if __name__ == "__main__":
    sys.exit(main())
