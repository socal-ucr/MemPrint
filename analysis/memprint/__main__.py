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

    if args.action == "windowed":
        results, curves = [], []
        for w in workloads:
            directory = Path(args.run) if args.run else paths.traces / w
            r, c = timeline.evaluate_windowed(traces.load_timelines(directory, prefixes=("Buffered", "Spatial")),
                                              traces.load_windowed(directory), w)
            results.append(r)
            curves.append(c)
        results = pd.concat(results)
        write(results, paths.data / "timeline_windowed.csv", index=False)
        write(pd.concat(curves), paths.data / "timeline_windowed_curves.csv", index=False)
        pd.set_option("display.width", 200)
        print(results.pivot_table(index=["workload", "config", "sample_rate", "watched"], columns="estimate",
                                  values=["mape", "peak_error"]).round(1).to_string())
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
    columns = [c for c in ["splitter_mape", "splitter_peak_error", "splitter_error_at_peak", "sampler_mape",
                           "sampler_peak_error", "sampler_error_at_peak", "sampler_length_error"] if c in recon]
    print("(peak_error: error of the estimated peak; error_at_peak: error when the true footprint peaks)")
    print(recon.groupby(["split", "subset", "variant"])[columns].agg(lambda x: np.mean(np.abs(x))).round(2).to_string())
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


def _spectrum_job(polybench, workload, config, out, footprint):
    from . import static

    spec, result, errors, seconds = static.analyze_polybench(polybench, workload, config, footprint)
    static.save(out, spec, result, errors, seconds)
    return f"{workload} {config}: {seconds:.1f}s footprint {spec.footprint:.0f} B, coverage {result.coverage:.3f}"


def cmd_static(args, paths):
    from concurrent.futures import ProcessPoolExecutor

    from . import lowo, static
    from .static import idioms, runs

    out = paths.data / "static"
    workloads = args.workloads or [w for w in default_workloads() if w != "miniVite"]
    if args.action == "spectra":
        jobs = [(w, c) for w in workloads for c in static.POLYBENCH_CONFIGS
                if args.force or not (out / f"{w}-{c}.npz").exists()]
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(_spectrum_job, args.polybench, w, c, out / f"{w}-{c}.npz", args.footprint)
                       for w, c in jobs]
            for (w, c), f in zip(jobs, futures):
                try:
                    print(f.result(), flush=True)
                except Exception as e:  # a config the kernel does not define fails to parse
                    print(f"{w} {c}: failed ({e})", flush=True)
        return

    if args.action == "idioms":
        rows = {k: idioms.features([src], defines=["MEDIUM_DATASET"], includes=[Path(args.polybench) / "utilities"])
                for k, src in static.polybench_sources(args.polybench).items()}
        for spec in args.program or []:  # name=root:file[,file...]
            name, rest = spec.split("=", 1)
            root, files = rest.split(":", 1)
            rows[name] = idioms.features(files.split(","), root=root, includes=args.include or [])
        table = pd.DataFrame(rows).T
        table["out_of_distribution"] = [idioms.out_of_distribution(r) for _, r in table.iterrows()]
        write(table, paths.data / "static_idioms.csv")
        print(table.round(3).to_string())
        return

    if args.action == "gap":
        from . import irregular

        kernels = [k for k in irregular.KERNELS if paths.all_data(k).exists()]
        if args.source == "interp":                                    # serial workloads only
            kernels = [k for k in kernels if runs.GAP_KERNEL.match(k) and (not args.workloads or k in args.workloads
                                                                          or any(k in irregular.BASELINE_FROM[w]
                                                                                 for w in args.workloads
                                                                                 if w in irregular.BASELINE_FROM))]
        all_data = {k: paths.read_all_data(k) for k in kernels}
        scales = {k: sorted(all_data[k]["Config"].astype(str).unique(), key=int) for k in kernels}
        if args.scales:
            scales = {k: [s for s in v if s in args.scales] for k, v in scales.items()}
            all_data = {k: d[d["Config"].astype(str).isin(scales[k])] for k, d in all_data.items()}
        if args.source == "interp":
            spectra = {}
            for k in kernels:
                spectra.update(runs.gap_spectra([k], scales[k], args.gapbs, paths.data / "static_auto", args.jobs))
            all_data = {k: d[d["Config"].astype(str).isin(spectra[k])] for k, d in all_data.items()}
            missing = [k for k, d in all_data.items() if d.empty]
            if missing:
                print("no spectra (interpreter failed) for:", " ".join(missing))
            all_data = {k: d for k, d in all_data.items() if not d.empty}
            kernels = [k for k in kernels if k in all_data]
        else:
            spectra = {k: irregular.skeleton_spectra(k, scales[k], paths.data / "static_gap") for k in kernels}
        kernels = [k for k in kernels if all(b in all_data for b in irregular.BASELINE_FROM[k])]
        borrow = {}
        if args.borrow:
            borrow = {w: pd.read_csv(Path(args.borrow) / f"{w}_allData.csv")
                      for w in default_workloads() if w != "miniVite" and (Path(args.borrow) / f"{w}_allData.csv").exists()}
        targets = [k for k in kernels if not args.workloads or k in args.workloads]
        errors, footprints = irregular.evaluate(all_data, spectra, borrow, kernels=targets)
        if args.workloads:
            errors = errors[errors["kernel"].isin(args.workloads)]
            footprints = footprints[footprints["kernel"].isin(args.workloads)]
        stem = "gap_pilot" if args.source == "skeleton" else "gap_interp"
        write(errors, paths.data / f"{stem}_errors.csv", index=False)
        write(footprints, paths.data / f"{stem}_footprints.csv", index=False)
        pd.set_option("display.width", 200)
        print(footprints.round(2).to_string())
        print(errors.pivot_table(index=["kernel", "split"], columns="method", values="mape").round(2).to_string())
        return

    if args.action == "programs":
        # C programs the interpreter runs by itself (tests/static/*.c, traced as <name>-<SCALE>);
        # the runtime baseline is fitted on the PolyBench traces and spectra under --borrow's root.
        from . import irregular

        programs = args.workloads or ["csr_pr", "csr_bfs"]
        all_data = {k: paths.read_all_data(k) for k in programs}
        spectra = {}
        for k in programs:
            spectra[k] = {}
            for config in sorted(all_data[k]["Config"].astype(str).unique(), key=int):
                path = out / f"{k}-{config}.npz"
                if not path.exists():
                    spec, result, errors, seconds = static.analyze(
                        Path(__file__).resolve().parents[2] / "tests" / "static" / f"{k}.c", [f"SCALE={config}"],
                        footprint="bytes")
                    static.save(path, spec, result, errors, seconds)
                spectra[k][config] = static.load(path)[0]
        pb = Path(args.borrow)
        pb_spectra = {tuple(f.stem.rsplit("-", 1)): static.load(f)[0] for f in sorted((pb / "static").glob("*.npz"))}
        pb_data = {w: pd.read_csv(pb / f"{w}_allData.csv") for w in sorted({w for w, _ in pb_spectra})}
        baseline = lowo.fit_baseline(list(lowo.load(pb_data, pb_spectra).values()))
        errors, footprints = irregular.evaluate(all_data, spectra, pb_data, {k: baseline for k in programs})
        write(errors, paths.data / "programs_errors.csv", index=False)
        write(footprints, paths.data / "programs_footprints.csv", index=False)
        pd.set_option("display.width", 200)
        print(footprints.round(2).to_string())
        print(errors.pivot_table(index=["kernel", "split"], columns="method", values="mape").round(2).to_string())
        return

    if args.action == "cpp":
        # any C/C++ program: python -m memprint static cpp NAME --source FILE --configs ... --args ...
        from . import irregular, similarity as sim
        from .dataset import prepare

        if not args.workloads or not args.source_file:
            raise SystemExit("static cpp NAME --source FILE --configs C1 C2 ... [--args ARG ...]")
        name = args.workloads[0]
        import shlex
        cmdline = shlex.split(args.argline) if args.argline else (args.args or [])
        spectra, meta = runs.program_spectra(name, args.source_file, args.configs, cmdline,
                                             args.define or [], args.include or [], heap_top=args.heap_top,
                                             cache_dir=paths.data / "static_cpp", n_jobs=args.jobs)
        base_root = Path(args.baseline_root or paths.data)
        sources = []
        for w in args.baseline_from or []:
            a = pd.read_csv(base_root / f"{w}_allData.csv").assign(Config=lambda d: d["Config"].astype(str))
            specs = {f.stem.rsplit("-", 1)[1]: static.load(f)[0]
                     for f in sorted((base_root / args.baseline_spectra).glob(f"{w}-*.npz"))}
            sources.append((prepare(a), specs, sim.bin_summary(a)))
        baseline = irregular.fit_baseline(sources) if sources else None
        if baseline is None:
            from .static.spectrum import BASIS, Baseline
            baseline = Baseline(np.zeros(len(BASIS)), np.zeros(len(BASIS)))
        pred = runs.predictions(name, spectra, meta, baseline)
        write(pred, paths.data / f"{name}_predicted.csv", index=False)
        print(pred[pred.k == 1][["config", "footprint", "program_footprint", "coverage", "seconds"]].to_string())
        if paths.all_data(name).exists():
            scored = runs.score(pred, paths.read_all_data(name))
            write(scored, paths.data / f"{name}_scored.csv", index=False)
            print(scored.round(2).to_string())
        return

    # lowo: leave one workload out
    spectra = {}
    for f in sorted(out.glob("*.npz")):
        w, c = f.stem.rsplit("-", 1)
        if w in workloads:
            spectra[(w, c)] = static.load(f)[0]
    all_data = {w: paths.read_all_data(w) for w in sorted({w for w, _ in spectra})}
    ast = None
    if args.ast:
        ast = pd.read_csv(args.ast)
        ast.index = ast.pop("file").str.replace(".c.ast.json", "", regex=False)
        ast = np.log1p(ast.astype(float))
    known = lowo.load(all_data, spectra)
    errors, validity, descriptors = lowo.evaluate(known, ast)
    write(errors, paths.data / "lowo_errors.csv", index=False)
    write(validity, paths.data / "lowo_validity.csv", index=False)
    write(descriptors, paths.data / "lowo_descriptors.csv")
    checks = [lowo.transfer_check(w, paths.read_all_data(w), known) for w in args.transfer or []]
    if checks:
        write(pd.DataFrame(checks), paths.data / "lowo_transfer.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMAPE (%) of alpha on the held-out workload's test config:")
    print(lowo.summarize(errors).round(2).to_string())
    print("\nSpearman rho between descriptor distance and transfer error (median over held-out workloads):")
    print(validity.groupby("split")[["rho_static", "rho_measured", "rho_ast", "zhat_reuse_rmse"]].median()
          .round(3).to_string())
    for check in checks:
        print(f"\n{check}")
    from .plots.static import plot_lowo

    plot_lowo(errors, validity, descriptors, paths.figures / "static")


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
    p.add_argument("action", choices=["preprocess", "build", "estimate", "forecast", "windowed"],
                   help="preprocess: traces/<wl>/*_timeline.csv -> data/<wl>_timeline.csv; "
                        "build: evaluate and fit models -> data/timeline_*.csv; "
                        "estimate/forecast: apply the model to sampler runs; "
                        "windowed: evaluate -window runs against the splitter -> data/timeline_windowed.csv")
    p.add_argument("workloads", nargs="*")
    p.add_argument("--subset", choices=["NZ", "MT", "L2O"], default="L2O", help="build: training subset of the final model")
    p.add_argument("--variant", choices=list(timeline.VARIANTS), default="base",
                   help="build: model features (time adds log Time)")
    p.add_argument("--run", help="estimate/forecast/windowed: directory with the timelines (default traces/<wl>)")
    p.add_argument("--upto", type=float, help="forecast: use the run up to this many memory references")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("static", help="static analysis of workload sources and leave-one-workload-out evaluation")
    p.add_argument("action", choices=["spectra", "idioms", "lowo", "gap", "programs", "cpp"],
                   help="spectra: PolyBench access-count spectra -> data/static/<wl>-<config>.npz; "
                        "idioms: access-idiom features -> data/static_idioms.csv; "
                        "lowo: predict each workload from its source and the others -> data/lowo_*.csv; "
                        "gap: GAP pilot, predict the GAP workloads from skeletons -> data/gap_pilot_*.csv; "
                        "programs: C programs in tests/static run by the interpreter -> data/programs_*.csv")
    p.add_argument("workloads", nargs="*", help="default: the PolyBench kernels in workloads.txt")
    p.add_argument("--polybench", help="PolyBench/C source tree (spectra, idioms)")
    p.add_argument("--jobs", type=int, default=8, help="spectra: parallel processes")
    p.add_argument("--force", action="store_true", help="spectra: recompute existing spectra")
    p.add_argument("--footprint", choices=["bytes", "starts"], default="bytes",
                   help="spectra: footprint definition of the traces they are compared with (default bytes)")
    p.add_argument("--program", nargs="*", help="idioms: more programs as name=root:file[,file...]")
    p.add_argument("--include", nargs="*", help="idioms: include directories for --program")
    p.add_argument("--ast", help="lowo: clang AST node counts per kernel (CSV) for the AST-similarity baseline")
    p.add_argument("--transfer", nargs="*", help="lowo: workloads without a spectrum to check borrowed models on")
    p.add_argument("--borrow", help="gap, programs: directory of PolyBench allData tables (and static/ spectra)")
    p.add_argument("--source", choices=["skeleton", "interp"], default="skeleton",
                   help="gap: hand-written skeleton (data/static_gap) or the C++ interpreter on GAP's source "
                        "(data/static_auto; results in data/gap_interp_*.csv)")
    p.add_argument("--gapbs", default="workloads/src/gapbs", help="gap --source interp: GAP source tree")
    p.add_argument("--scales", nargs="*", help="gap: only these scales")
    p.add_argument("--source-file", dest="source_file",
                   help="cpp: the program's source file, or several comma-separated (compiled as one unity file)")
    p.add_argument("--configs", nargs="*", default=[], help="cpp: input configs")
    p.add_argument("--args", nargs="*", help="cpp: command line after the program name; {config} is replaced")
    p.add_argument("--argline", help="cpp: the same as one string (for arguments that start with '-')")
    p.add_argument("--define", nargs="*", help="cpp: preprocessor definitions (NAME=VALUE); {config} is replaced")
    p.add_argument("--heap-top", dest="heap_top", type=int, default=0,
                   help="cpp: free bytes in glibc's top chunk at the first large allocation")
    p.add_argument("--baseline-from", dest="baseline_from", nargs="*",
                   help="cpp: workloads of the same runtime whose traces and spectra fit the runtime baseline")
    p.add_argument("--baseline-root", dest="baseline_root", help="cpp: directory with their allData tables")
    p.add_argument("--baseline-spectra", dest="baseline_spectra", default="static_auto",
                   help="cpp: subdirectory of --baseline-root with their spectra")
    p.set_defaults(func=cmd_static)

    p = sub.add_parser("paper-figures", help="regenerate the paper's data figures into figures/paper/")
    p.add_argument("--massif-csv", help="CSV path, may contain {workload} (default results/<wl>/massif.csv)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_paper_figures)

    args = parser.parse_args(argv)
    args.func(args, Paths(args.root))


if __name__ == "__main__":
    sys.exit(main())
