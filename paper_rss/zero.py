"""A prediction with no run of the program: the interpreter's footprint from source plus terms learned from the
RSS runs of other programs (paper_rss, the generalisation section).

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/zero.py   ->  paper_rss/data/zero.csv, tables/zero.tex

Programs: the 11 corpus programs the interpreter runs at the largest input (zero_static.py): PolyBench 2mm, atax,
gemm and jacobi-2d at LARGE (the live footprint over the interpreter's reference clock, taken as the progress axis),
and GAP bfs, cc, pr and sssp at scale 18, HPCCG at n = 64, LULESH at s = 60 and NPB MG at W (the footprint of the
spectrum, held over the whole run: the C++ interpreter has no timeline mode). P(t) is the program's predicted
footprint on the 100 steps.

Two targets:
  footprint  the truth of compute.truth (pages in use plus the runtime's pages B). Predicted: P(t) + R, with R the
             median of the other programs' B (their runtime's pages, from their RSS runs).
  RSS        the RSS of a plain native run of the program (the earlier run of compute.model). Predicted:
             P(t) / (1 - r) + R', with r the median over the other programs of the allocator's free share of
             anonymous RSS (free / anonymous at the model run's peak) and R' the median over the other programs of
             their RSS peak less their largest data estimate A - F (the libraries' pages and the runtime's), both from
             the other programs' runs.
Changed after the first results: R and R' were the medians of (truth peak - P peak) and (RSS peak - P peak) over the
other programs, which absorbs the interpreter's own errors on them (GAP's footprint counts the bytes its graph
builder frees, 1.4x the pages in use) and gave R = -15 MB; they now come from the other programs' runs alone.
Both are compared with the RSS model's own run (estimate 1), its transfer from the smaller inputs, MemPrint W5 and
the truth's own peak held flat. Leave one program out for R, r and R'.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute as M  # noqa: E402
from memprint.sched import corpus as C  # noqa: E402
from memprint.sched.run import largest  # noqa: E402

Z = os.path.expanduser("~/memprint-data/2026-10-28-rsszero")
ROOT = os.path.join(HERE, "..")
POLY = ("2mm", "atax", "gemm", "jacobi-2d")
SPECTRA = {**{f"gap-{k}": f"{ROOT}/data/gap-bytes/data/static_auto/gap_{k}_kron-18.npz" for k in ("bfs", "cc", "pr", "sssp")},
           "npb-mg": f"{ROOT}/data/blind2/data/static_cpp/npb_mg-W.npz",
           "hpccg": f"{Z}/hpccg.npz", "lulesh": f"{Z}/lulesh.npz"}


def footprint_of(path):
    d = np.load(path)
    if "footprint" in d.files:
        return float(d["footprint"])
    # a spectrum (paper_generalization's): n addresses of s bytes per group (memprint.static.Spectrum.footprint)
    return float(np.sum(d["n"] * d["s"]))


def program_curve(prog, n):
    if prog in POLY:
        d = np.load(f"{Z}/{prog}.npz")
        if "u" in d.files:   # timeline mode (atax; the others ran out of memory in it, footprint alone)
            return C.resample(d["u"], d["live"], n), float(d["seconds"])
        return np.full(n, float(d["footprint"])), float(d["seconds"])
    d = np.load(SPECTRA[prog])
    secs = float(d["seconds"]) if "seconds" in d.files else np.nan
    return np.full(n, footprint_of(SPECTRA[prog])), secs


def main():
    ins = M.inputs()
    progs = [p for p in list(POLY) + list(SPECTRA) if os.path.exists(SPECTRA.get(p, f"{Z}/{p}.npz"))]
    cur = largest(pd.read_csv(f"{M.D}/study/curves.csv", dtype={"input": str}))
    mc = pd.read_csv(f"{M.D}/model/curves.csv", dtype={"input": str})
    info = {}
    for p in progs:
        inp = ins[p]
        t, base = M.truth(p, inp), M.base_of(p, inp)
        P, secs = program_curve(p, len(t))
        runs = []
        for rep in (1, 2, 3):
            o = f"{M.O}/runs_all/{p}/{rep}"
            b = pd.read_csv(f"{o}/base2.csv"); b = b[b.rss_bytes > 0]
            r = pd.read_csv(f"{o}/pss.csv"); r = r[r.rss_bytes > 0]
            k = int(np.argmax(r.anon_bytes))
            runs.append(dict(rss=C.resample(b.seconds, b.rss_bytes, len(t)),
                             est1=C.resample(r.seconds, np.clip(r.anon_bytes - r.free_bytes, 0, None) + base, len(t)),
                             free_share=r.free_bytes.iloc[k] / max(r.anon_bytes.iloc[k], 1),
                             rest=b.rss_bytes.max() - (r.anon_bytes - r.free_bytes).max()))
        w5 = cur[(cur.prog == p) & (cur.input == inp) & (cur.method == "W5nat")].sort_values("k").bytes.to_numpy()
        x = mc[(mc.prog == p) & (mc.input == inp) & (mc.method == "model1X")].sort_values("k").bytes.to_numpy()
        info[p] = dict(t=t, base=base, P=P, secs=secs, runs=runs, w5=w5 + base if len(w5) else None,
                       X=x + base if len(x) else None)
    rows = []
    for p, d in info.items():
        others = [q for q in info if q != p]
        R = np.median([info[q]["base"] for q in others])
        rr = np.median([np.median([u["free_share"] for u in info[q]["runs"]]) for q in others])
        Rp = np.median([np.median([u["rest"] for u in info[q]["runs"]]) for q in others])
        t, P = d["t"], d["P"]
        est = {"zero": P + R, "flat_truth_peak": np.full(len(t), t.max()), "est1": d["runs"][0]["est1"]}
        if d["X"] is not None:
            est["transfer"] = d["X"]
        if d["w5"] is not None:
            est["memprint"] = d["w5"]
        for m, e in est.items():
            rows.append(dict(prog=p, target="footprint", method=m, **M.err(e, t), R_mb=R / 1e6, secs=d["secs"]))
        for k, u in enumerate(d["runs"]):
            rt = u["rss"]
            for m, e in (("zero", P / (1 - rr) + Rp), ("est1", d["runs"][(k + 1) % 3]["est1"]),
                         ("rss_other_run", d["runs"][(k + 1) % 3]["rss"])):
                rows.append(dict(prog=p, target="rss", method=m, rep=k, **M.err(e, rt), R_mb=Rp / 1e6, r=rr))
    out = pd.DataFrame(rows)
    out = out.groupby(["prog", "target", "method"]).median(numeric_only=True).reset_index()
    os.makedirs(f"{HERE}/data", exist_ok=True)
    out.to_csv(f"{HERE}/data/zero.csv", index=False)
    pd.set_option("display.width", 220)
    for tg in ("footprint", "rss"):
        o = out[out.target == tg]
        print(tg); print(o.pivot(index="prog", columns="method", values=["signed", "peak"]).round(1).to_string())
        print(o.groupby("method")[["signed", "abs", "peak"]].median().round(1))
    print(out[(out.target == "footprint") & (out.method == "zero")][["prog", "R_mb", "secs"]].round(2).to_string(index=False))
    print("r", out[out.target == "rss"].r.dropna().round(4).unique())
    table(out)


def table(out):
    """tables/zero.tex: per program, the interpreter's time over the native run's, the peak error and the signed
    whole-run error (% of the truth) of the prediction with no run, the model's own run, its transfer and
    MemPrint, and the zero-run prediction's peak error against the plain run's RSS."""
    nat = M.walls(f"{M.O}/done_all.txt")
    names = {"2mm": "2mm", "atax": "atax", "gemm": "gemm", "jacobi-2d": "jacobi-2d", "gap-bfs": "GAP bfs",
             "gap-cc": "GAP cc", "gap-pr": "GAP pr", "gap-sssp": "GAP sssp", "hpccg": "HPCCG", "lulesh": "LULESH",
             "npb-mg": "NPB MG"}
    f = out[out.target == "footprint"].set_index(["prog", "method"])
    r = out[out.target == "rss"].set_index(["prog", "method"])
    sg = lambda v: "--" if pd.isna(v) else f"{v:+.1f}".replace("-", "$-$")
    get = lambda d, p, m, c: d[c].get((p, m), np.nan)
    lines = []
    for p in names:
        if (p, "zero") not in f.index:
            continue
        x = get(f, p, "zero", "secs") / nat[(p, "base")]
        cells = [names[p], f"{x:,.0f}".replace(",", "{,}")]
        cells += [sg(get(f, p, m, "peak")) for m in ("zero", "est1", "transfer", "memprint")]
        cells += [sg(get(f, p, m, "signed")) for m in ("zero", "est1")]
        cells += [sg(get(r, p, "zero", "peak"))]
        lines.append(" & ".join(cells) + r" \\")
    open(f"{HERE}/tables/zero.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
