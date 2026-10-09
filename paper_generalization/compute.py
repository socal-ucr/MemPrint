"""Tables behind the figures of the generalisation paper, computed from the traces and spectra in data/.

python paper_generalization/compute.py   (writes paper_generalization/data/*.csv)

Every number is recomputed from the repository's data: PolyBench (data/polybench-bytes), the GAP
pilot (data/gap-bytes: hand skeleton spectra in static_gap, interpreter spectra in static_auto) and
the two C graph programs (data/csr-bytes). Nothing here is fitted on the workload it predicts: the
runtime baseline of a PolyBench kernel is fitted on the other 26, that of a GAP kernel on the
workloads in irregular.BASELINE_FROM.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from memprint import irregular, lowo  # noqa: E402
from memprint import similarity as sim  # noqa: E402
from memprint.dataset import prepare  # noqa: E402
from memprint.static import load as load_static  # noqa: E402
from memprint.static.spectrum import BASIS, Spectrum, moments, sample_probability  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PB = ROOT / "data" / "polybench-bytes" / "data"
GAP = ROOT / "data" / "gap-bytes" / "data"
HERE_BLIND = Path(__file__).resolve().parent / "blind"


def write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df)} rows")


# ---------------------------------------------------------------- PolyBench

def polybench():
    spectra, meta = {}, []
    for f in sorted((PB / "static").glob("*.npz")):
        w, c = f.stem.rsplit("-", 1)
        spec, m = load_static(f)
        spectra[(w, c)] = spec
        meta.append({"workload": w, "config": c, "footprint": spec.footprint, "references": spec.references,
                     "addresses": spec.addresses, **m})
    meta = pd.DataFrame(meta)
    meta["coverage"] = 1 - (meta["unresolved"] + meta["uncertain"]) / meta["total"]
    pin = []
    for w in meta.workload.unique():
        a = pd.read_csv(PB / f"{w}_allData.csv")
        a = a[a.SamplingInterval == 1].assign(config=a.Config.astype(str).str.split("-").str[-1])
        pin.append(a.groupby("config").CountObs.min().rename("pin_references").reset_index().assign(workload=w))
    meta = meta.merge(pd.concat(pin), on=["workload", "config"], how="left")
    write(meta, "pb_static_meta.csv")
    data = {w: pd.read_csv(PB / f"{w}_allData.csv") for w in sorted({w for w, _ in spectra})}
    known = lowo.load(data, spectra)

    # per (kernel, config, k): measured bins against the moments of the static spectrum, with the
    # runtime baseline fitted on the other kernels (as in the leave-one-out evaluation)
    rows, bins = [], []
    for held, C in sorted(known.items()):
        baseline = lowo.fit_baseline([wl for w, wl in known.items() if w != held])
        truth_u = C.all_data[C.all_data["SamplingInterval"] == 1].groupby("Config")["UniqueAddresses"].min()
        for config, spec in C.spectra.items():
            if config not in C.prepared.truth:
                continue
            s = C.summary[C.summary["Config"] == config].set_index("SamplingInterval")
            for k in sim.INTERVALS:
                if k not in s.index:
                    continue
                bm = moments(spec, k, baseline)
                bs = moments(spec, k)
                rows.append({"workload": held, "config": config, "k": k, "truth": C.prepared.truth[config],
                             "truth_unique": float(truth_u.get(config, np.nan)),
                             "m_meas": s.loc[k, "m"], "sd_meas": s.loc[k, "sd"], "u_meas": s.loc[k, "u"],
                             "m_pred": bm.m, "sd_pred": bm.sd, "u_pred": bm.u, "truth_pred": bm.truth,
                             "alpha_pred": bm.alpha, "alpha_meas": C.prepared.truth[config] / s.loc[k, "m"],
                             "m_program": bs.m, "baseline_fp": baseline.footprint})
            d = C.prepared.data
            d = d[(d["Config"] == config) & d["SamplingInterval"].isin(sim.INTERVALS)]
            for _, r in d.iterrows():
                bins.append({"workload": held, "config": config, "k": int(r["SamplingInterval"]),
                             "m": r["MemUsageObs"], "alpha": r["Alpha"]})
    write(pd.DataFrame(rows), "pb_moments.csv")
    write(pd.DataFrame(bins), "pb_bins.csv")

    # cumulative spectra at MEDIUM: bytes referenced at most c times
    spec_rows = []
    for (w, c), spec in spectra.items():
        if c != "MEDIUM":
            continue
        order = np.argsort(spec.c)
        cum = np.cumsum((spec.n * spec.s)[order]) / spec.footprint
        cc = spec.c[order]
        grid = np.unique(np.round(np.logspace(0, np.log10(max(cc.max(), 2)), 120), 6))
        frac = np.interp(np.log(grid), np.log(np.maximum(cc, 1e-9)), cum, left=0, right=1)
        spec_rows += [{"workload": w, "c": g, "frac_bytes": f} for g, f in zip(grid, frac)]
    write(pd.DataFrame(spec_rows), "pb_cumulative_spectra.csv")

    # the runtime baseline fitted on all 27 kernels
    base = lowo.fit_baseline(list(known.values()))
    write(pd.DataFrame({"count": BASIS, "bytes": base.bytes, "addresses": base.addresses}), "pb_baseline.csv")
    return known


# ---------------------------------------------------------------- GAP

GAP_KERNELS = ["pr", "bfs", "cc", "bc", "tc", "sssp"]


def _gap_spectra(directory, kernel):
    out = {}
    for p in sorted((GAP / directory).glob(f"{kernel}-*.npz")):
        out[p.stem.rsplit("-", 1)[1]] = Spectrum.from_dict(np.load(p))
    return out


def gap():
    rows, curves = [], []
    hand_all = {}
    for base in GAP_KERNELS:
        for graph in ("uniform", "kron"):
            kernel = f"gap_{base}" + ("_kron" if graph == "kron" else "")
            if not (GAP / f"{kernel}_allData.csv").exists():
                continue
            srcs = []
            for b in irregular.BASELINE_FROM[kernel]:
                a = pd.read_csv(GAP / f"{b}_allData.csv").assign(Config=lambda x: x.Config.astype(str))
                srcs.append((prepare(a), _gap_spectra("static_gap", b), sim.bin_summary(a)))
            baseline = irregular.fit_baseline(srcs)
            a = pd.read_csv(GAP / f"{kernel}_allData.csv").assign(Config=lambda x: x.Config.astype(str))
            P, summary = prepare(a), sim.bin_summary(a)
            hand, auto = _gap_spectra("static_gap", kernel), _gap_spectra("static_auto", kernel)
            hand_all[kernel] = hand
            for scale in P.configs:
                d = P.data[(P.data.Config == scale) & P.data.SamplingInterval.isin(sim.INTERVALS)]
                alpha = d.Alpha.to_numpy(float)
                s = summary[summary.Config == scale].set_index("SamplingInterval")
                for source, specs in (("hand", hand), ("auto", auto)):
                    spec = specs.get(scale)
                    if spec is None:
                        continue
                    pred = d.SamplingInterval.map(lambda k: moments(spec, k, baseline).alpha).to_numpy()
                    truth = P.truth[scale]
                    rows.append({"kernel": base, "graph": graph, "scale": int(scale), "source": source,
                                 "truth": truth, "program_fp": spec.footprint, "baseline_fp": baseline.footprint,
                                 "fp_error": (spec.footprint + baseline.footprint - truth) / truth * 100,
                                 "alpha_mape": float(np.mean(np.abs(pred - alpha) / alpha) * 100),
                                 "references": spec.references})
                    for k in sim.INTERVALS:
                        if k in s.index:
                            bm = moments(spec, k, baseline)
                            curves.append({"kernel": base, "graph": graph, "scale": int(scale), "source": source,
                                           "k": k, "alpha_pred": bm.alpha, "alpha_meas": truth / s.loc[k, "m"],
                                           "sd_pred": bm.sd, "sd_meas": s.loc[k, "sd"], "m_pred": bm.m,
                                           "m_meas": s.loc[k, "m"]})
    write(pd.DataFrame(rows), "gap_scores.csv")
    write(pd.DataFrame(curves), "gap_curves.csv")

    # cumulative spectra at scale 12: hand skeleton against interpreter
    spec_rows = []
    for base in GAP_KERNELS:
        for graph in ("uniform", "kron"):
            kernel = f"gap_{base}" + ("_kron" if graph == "kron" else "")
            for source, d in (("hand", "static_gap"), ("auto", "static_auto")):
                p = GAP / d / f"{kernel}-12.npz"
                if not p.exists():
                    continue
                spec = Spectrum.from_dict(np.load(p))
                order = np.argsort(spec.c)
                cum = np.cumsum((spec.n * spec.s)[order]) / spec.footprint
                cc = spec.c[order]
                grid = np.unique(np.round(np.logspace(0, np.log10(max(cc.max(), 2)), 120), 6))
                frac = np.interp(np.log(grid), np.log(np.maximum(cc, 1e-9)), cum, left=0, right=1)
                spec_rows += [{"kernel": base, "graph": graph, "source": source, "c": g, "frac_bytes": f}
                              for g, f in zip(grid, frac)]
    write(pd.DataFrame(spec_rows), "gap_cumulative_spectra.csv")

    # runtime baseline of the GAP runs (fitted on pr and bfs, uniform)
    srcs = []
    for b in ("gap_pr", "gap_bfs"):
        a = pd.read_csv(GAP / f"{b}_allData.csv").assign(Config=lambda x: x.Config.astype(str))
        srcs.append((prepare(a), _gap_spectra("static_gap", b), sim.bin_summary(a)))
    base = irregular.fit_baseline(srcs)
    write(pd.DataFrame({"count": BASIS, "bytes": base.bytes, "addresses": base.addresses}), "gap_baseline.csv")


def interpreter_cost():
    """Seconds per GAP workload and scale of the final interpreter, from the log of
    `memprint static gap --source interp --jobs 6` (data/interp_times.txt; 6 runs in parallel)."""
    rows = []
    for line in (OUT / "interp_times.txt").read_text().splitlines():
        parts = line.replace(",", "").split()
        if len(parts) >= 7 and parts[1] == "footprint" and parts[-1] == "s":
            name, scale = parts[0].rstrip(":").rsplit("-", 1)
            k = name.replace("gap_", "").replace("_kron", "")
            rows.append({"kernel": k, "scale": int(scale), "graph": "kron" if "_kron" in name else "uniform",
                         "seconds": float(parts[-2]), "coverage": float(parts[5]), "parallel": 6})
    write(pd.DataFrame(rows), "interp_cost.csv")


def blind():
    """The blind test: predicted (committed) against measured alpha per interval and config."""
    B = ROOT / "data" / "blind" / "data"
    rows = []
    for prog in ("hpccg", "lulesh"):
        pred = pd.read_csv(HERE_BLIND / f"{prog}_predicted.csv").assign(config=lambda d: d.config.astype(str))
        a = pd.read_csv(B / f"{prog}_allData.csv").assign(Config=lambda d: d.Config.astype(str))
        P, summary = prepare(a), sim.bin_summary(a)
        for c in P.configs:
            s = summary[summary.Config == c].set_index("SamplingInterval")
            g = pred[pred.config == c].set_index("k")
            for k in sim.INTERVALS:
                if k in s.index:
                    rows.append({"program": prog, "config": int(c), "k": k, "alpha_pred": g.loc[k, "alpha"],
                                 "alpha_meas": P.truth[c] / s.loc[k, "m"]})
    write(pd.DataFrame(rows), "blind_curves.csv")


def degrees():
    """Degree distributions of the two generators at scale 14 (degree 16, symmetrised, no duplicates)."""
    from memprint.static import gap as g

    rows = []
    for graph, fn in (("uniform", g.uniform_edges), ("kron", g.rmat_edges)):
        out = fn(14, 16)
        n, u, v = out[0], out[1], out[2]
        pairs = np.unique(np.concatenate([np.stack([u, v], 1), np.stack([v, u], 1)]), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        deg = np.bincount(pairs[:, 0], minlength=n)
        values, counts = np.unique(deg, return_counts=True)
        rows += [{"graph": graph, "degree": int(d), "vertices": int(c)} for d, c in zip(values, counts)]
    write(pd.DataFrame(rows), "gap_degrees.csv")


def inclusion():
    """p(c, k) = 1 - (1 - 1/k)^c for the figure on inclusion probabilities."""
    c = np.unique(np.round(np.logspace(0, 6, 200)))
    rows = [{"k": k, "c": x, "p": float(sample_probability(x, k))} for k in (100, 1000, 10000, 100000) for x in c]
    write(pd.DataFrame(rows), "inclusion.csv")


if __name__ == "__main__":
    inclusion()
    polybench()
    gap()
    degrees()
    interpreter_cost()
    blind()
    (OUT / "provenance.json").write_text(json.dumps({"root": str(ROOT)}, indent=1))
