"""Cost tables for every method in the paper, from the measurements of costs_pin.sh, costs_pin_train.sh and costs.py
(python paper_generalization/costs_tables.py).

Reads data/costs/{results,results_train}/<bench>/overhead_<mode>.csv (run.sh), data/costs/csr.csv, data/costs/static.csv
and the run times recorded in the spectra (GAP, HPCCG, LULESH, NPB). Writes
    data/pb_timings_measured.csv   PolyBench times in overhead.py's format (re-measured; interpreter = symbolic)
    data/costs_all.csv             one row per method and program set
    tables/costs_all.tex           the table of every method's cost
    tables/costs_steps.tex         the analysis steps (milliseconds)
    tables/costs_verify.tex        re-measured against the times used before
    tables/numbers_costs.tex       numbers quoted in the text
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C = ROOT / "data" / "costs"
T = HERE / "tables"
D = HERE / "data"
PBK = sorted({Path(f).stem.rsplit("-", 1)[0] for f in glob.glob(str(ROOT / "data/polybench-bytes/data/static/*.npz"))})
GAPK = [f"gap_{k}{g}" for g in ("", "_kron") for k in ("pr", "bfs", "cc", "bc", "tc", "sssp")]


def runsh(bench, mode, train=False):
    p = C / ("results_train" if train else "results") / bench / f"overhead_{mode}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["Config"] = d["Config"].astype(str)
    return d.drop_duplicates(["Config"], keep="last").set_index("Config")


def static():
    p = C / "static.csv"
    return pd.read_csv(p, dtype={"config": str}) if p.exists() else pd.DataFrame(
        columns=["part", "workload", "config", "step", "seconds", "reps", "load"])


def recorded(directory, pattern):
    out = {}
    for f in glob.glob(str(directory / pattern)):
        z = np.load(f)
        if "seconds" in z.files:
            out[Path(f).stem] = float(z["seconds"])
    return out


def pin_set(benches, configs):
    """Per (bench, config): native, instr, sampler, splitter seconds (NaN where missing)."""
    rows = []
    for b in benches:
        nat, ins, smp, spl = (runsh(b, m) for m in ("native", "instr", "sampler", "splitter"))
        for c in configs:
            g = lambda d, col: float(d.loc[c, col]) if c in d.index and pd.notna(d.loc[c, col]) else np.nan
            rows.append({"bench": b, "config": c, "native": g(nat, "Native(s)"), "instr": g(ins, "Pin(s)"),
                         "sampler": g(smp, "Pin(s)"), "splitter": g(spl, "Pin(s)")})
    return pd.DataFrame(rows)


def main():
    S = static()
    st = lambda part, step, cfg=None: S[(S.part == part) & (S.step == step) & ((S.config == cfg) if cfg else True)]
    rows = []

    def add(group, method, runs, seconds, native, trace, sampled, note=""):
        rows.append({"group": group, "method": method, "runs": runs, "seconds": seconds,
                     "x_native": seconds / native if native else np.nan,
                     "x_trace": seconds / trace if trace else np.nan,
                     "x_sampled": seconds / sampled if sampled else np.nan, "note": note})

    # ---------------- PolyBench at MEDIUM (medians over the 27 kernels of per-kernel ratios)
    pb = pin_set(PBK, ["SMALL2", "MEDIUM"])
    train = {b: runsh(b, "splitter", train=True) for b in PBK}
    pbm = pb[pb.config == "MEDIUM"].set_index("bench")
    sym = st("pb_interp", "interpreter (symbolic)", "MEDIUM").set_index("workload").seconds
    enu = st("pb_interp", "interpreter (enumerated)", "MEDIUM").set_index("workload").seconds
    tl = st("pb_interp", "interpreter (timeline)", "MEDIUM").set_index("workload").seconds
    parse = st("pb_interp", "clang parse", "MEDIUM").set_index("workload").seconds
    astc = st("pb_interp", "AST node counts", "MEDIUM").set_index("workload").seconds
    idi = st("pb_interp", "access-idiom features", "MEDIUM").set_index("workload").seconds

    def med(series, denom=None):
        s = series.reindex(pbm.index)
        return float((s / denom.reindex(pbm.index)).median()) if denom is not None else float(s.median())

    nat, trc, smp = pbm.native, pbm.splitter, pbm.sampler
    for name, ser, runs in [("Native run", nat, "--"), ("Pin, instrumentation only", pbm.instr, "--"),
                            ("Sampled run (\\memprint{}'s deployed run)", smp, "sampled run"),
                            ("Full trace (Pin splitter; exact footprint)", trc, "full trace")]:
        rows.append({"group": "PolyBench (27 kernels, MEDIUM)", "method": name, "runs": runs, "seconds": med(ser),
                     "x_native": med(ser, nat), "x_trace": med(ser, trc), "x_sampled": med(ser, smp), "note": ""})
    if len(sym):
        steps_ms = S[S.part == "pb_analysis"].groupby("step").seconds.median() * 1000
        moments_s = steps_ms.get("moments at 13 intervals (static alpha)", np.nan) / 1000
        for name, ser, runs in [
                ("Static footprint: C interpreter, symbolic counting", sym, "interpreter"),
                ("C interpreter, every iteration enumerated", enu, "interpreter"),
                ("Footprint over time: C interpreter, timeline mode", tl, "interpreter"),
                ("Static $\\alpha$, pooled regression, nearest/RBF by $\\hat z$", sym + moments_s + smp, "interpreter + sampled run"),
                ("Nearest model by AST counts", parse + astc + smp, "parse + sampled run"),
                ("Access-idiom gate (features)", idi, "parse"),
                ("Uniform mixture", smp, "sampled run")]:
            if ser.dropna().empty:
                continue
            rows.append({"group": "PolyBench (27 kernels, MEDIUM)", "method": name, "runs": runs, "seconds": med(ser),
                         "x_native": med(ser, nat), "x_trace": med(ser, trc), "x_sampled": med(ser, smp), "note": ""})
    own = pd.Series({b: sum(float(train[b].loc[c, "Pin(s)"]) for c in train[b].index) +
                     (float(pb[(pb.bench == b) & (pb.config == "SMALL2")].splitter.iloc[0]) if len(pb) else 0)
                     for b in PBK if len(train[b]) == 5})
    if len(own):
        tot = own + smp.reindex(own.index)
        rows.append({"group": "PolyBench (27 kernels, MEDIUM)", "method": "Own model (\\memprint{}): six training traces",
                     "runs": "6 full traces + sampled run", "seconds": float(tot.median()),
                     "x_native": float((tot / nat.reindex(own.index)).median()),
                     "x_trace": float((tot / trc.reindex(own.index)).median()),
                     "x_sampled": float((tot / smp.reindex(own.index)).median()), "note": ""})
        rows.append({"group": "PolyBench (27 kernels, MEDIUM)", "method": "Measured $z$ (nearest, RBF), oracle",
                     "runs": "full trace (+ sampled run)", "seconds": float((trc + smp).median()),
                     "x_native": float(((trc + smp) / nat).median()), "x_trace": float(((trc + smp) / trc).median()),
                     "x_sampled": float(((trc + smp) / smp).median()), "note": ""})

    # ---------------- GAP, 12 serial workloads
    gp = pin_set(GAPK, ["14", "16", "18"])
    rec = recorded(ROOT / "data/gap-bytes/data/static_auto", "*.npz")
    serial = st("cpp", "C++ interpreter", "16").set_index("workload").seconds
    skel = S[(S.part == "gap_skeleton")]
    for scale in ("16", "18"):
        g = gp[gp.config == scale].set_index("bench")
        if g.native.dropna().empty:
            continue
        grp = f"GAP (12 workloads, scale {scale})"
        n, tr, sa = g.native, g.splitter, g.sampler
        def m(ser):
            ser = ser.reindex(g.index)
            return (float(ser.median()), float((ser / n).median()),
                    float((ser / tr).median()) if tr.notna().any() else np.nan, float((ser / sa).median()))
        cand = [("Native run", n, "--"), ("Pin, instrumentation only", g.instr, "--"),
                ("Sampled run", sa, "sampled run"), ("Full trace", tr, "full trace"),
                ("Hand skeleton and its spectrum", skel[skel.config == scale].set_index("workload").seconds, "skeleton"),
                ("C++ interpreter (recorded, 4--16 jobs at a time)", pd.Series({k: rec.get(f"{k}-{scale}", np.nan) for k in GAPK}),
                 "interpreter")]
        if scale == "16" and len(serial):
            cand.append(("C++ interpreter (one at a time)", serial, "interpreter"))
        for name, ser, runs in cand:
            if ser.dropna().empty:
                continue
            s_, xn, xt, xs = m(ser)
            rows.append({"group": grp, "method": name, "runs": runs, "seconds": s_, "x_native": xn, "x_trace": xt,
                         "x_sampled": xs, "note": ""})

    # ---------------- the C programs, the blind tests
    if (C / "csr.csv").exists():
        cs = pd.read_csv(C / "csr.csv", dtype={"scale": str}).groupby(["program", "scale", "mode"]).seconds.median().unstack()
        ci = st("csr", "C interpreter").set_index(["workload", "config"]).seconds
        for (prog, scale), r in cs.iterrows():
            grp = "C program " + prog.replace('_', '\\_') + f" (scale {scale})"
            for name, v, runs in [("Native run", r.get("native"), "--"), ("Pin, instrumentation only", r.get("instr"), "--"),
                                  ("Sampled run", r.get("sampler"), "sampled run"), ("Full trace", r.get("splitter"), "full trace"),
                                  ("C interpreter", ci.get((prog, scale), np.nan), "interpreter")]:
                if pd.notna(v):
                    add(grp, name, runs, float(v), r.get("native"), r.get("splitter"), r.get("sampler"))
    blind = {"hpccg": (ROOT / "data/blind/data/static_cpp", ["20", "28"]), "lulesh": (ROOT / "data/blind/data/static_cpp", ["15", "20"]),
             "npb_is": (ROOT / "data/blind2/data/static_cpp", ["S", "W"]), "npb_cg": (ROOT / "data/blind2/data/static_cpp", ["S", "W"]),
             "npb_mg": (ROOT / "data/blind2/data/static_cpp", ["S", "W", "A"])}
    for b, (d, cfgs) in blind.items():
        p = pin_set([b], cfgs)
        p.loc[(p.bench == "npb_mg") & (p.config == "A"), "splitter"] = np.nan   # Pin ran out of memory (62 GB)
        rc = recorded(d, f"{b}-*.npz")
        for _, r in p.iterrows():
            grp = f"{b.replace('_', ' ').upper()} ({r.config})"
            for name, v, runs in [("Native run", r.native, "--"), ("Pin, instrumentation only", r.instr, "--"),
                                  ("Sampled run", r.sampler, "sampled run"), ("Full trace", r.splitter, "full trace"),
                                  ("C++ interpreter (recorded)", rc.get(f"{b}-{r.config}", np.nan), "interpreter")]:
                if pd.notna(v):
                    add(grp, name, runs, float(v), r.native, r.splitter, r.sampler)

    out = pd.DataFrame(rows)
    out.to_csv(D / "costs_all.csv", index=False)
    print(out.round(3).to_string())
    return out, S


# ---------------------------------------------------------------- re-measured PolyBench timings, verification

PB_CONFIGS = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]


def measured_pb_timings(S):
    """data/pb_timings_measured.csv in overhead.py's format, from the new measurements only."""
    rows = []
    for b in PBK:
        spl = pd.concat([runsh(b, "splitter"), runsh(b, "splitter", train=True)])
        for c in PB_CONFIGS:
            if c in spl.index:
                rows.append({"workload": b, "config": c, "mode": "splitter", "native_s": float(spl.loc[c, "Native(s)"]),
                             "pin_s": float(spl.loc[c, "Pin(s)"])})
        for mode in ("sampler", "instr"):
            d = runsh(b, mode)
            for c in ("SMALL2", "MEDIUM"):
                if c in d.index:
                    rows.append({"workload": b, "config": c, "mode": mode, "native_s": float(d.loc[c, "Native(s)"]),
                                 "pin_s": float(d.loc[c, "Pin(s)"])})
        for step, mode in (("interpreter (symbolic)", "interpreter"), ("clang parse", "clang parse")):
            d = S[(S.part == "pb_interp") & (S.step == step) & (S.workload == b)]
            for _, r in d.iterrows():
                rows.append({"workload": b, "config": r.config, "mode": mode, "native_s": np.nan, "pin_s": r.seconds})
    out = pd.DataFrame(rows)
    complete = out.groupby("workload").apply(lambda d: (d["mode"] == "splitter").sum() == 7 and
                                              {"sampler", "instr", "interpreter", "clang parse"} <= set(d["mode"]))
    if complete.all() and len(complete) == len(PBK):
        out.to_csv(D / "pb_timings_measured.csv", index=False)
    return out, bool(complete.all() and len(complete) == len(PBK))


def verify(S, new):
    """Re-measured PolyBench times against data/pb_timings.csv (the times used before), and the C++ interpreter
    run alone against the times recorded in GAP's spectra."""
    old = pd.read_csv(D / "pb_timings.csv")
    m = new.merge(old, on=["workload", "config", "mode"], suffixes=("_new", "_old"))
    rows = []
    for (mode, cfg), d in m.groupby(["mode", "config"]):
        if cfg not in ("SMALL2", "MEDIUM"):
            continue
        r = (d.pin_s_new / d.pin_s_old)
        rows.append({"what": {"splitter": "Full trace", "sampler": "Sampled run", "instr": "Instrumentation only",
                              "interpreter": "C interpreter (before: enumerated; now: symbolic)",
                              "clang parse": "clang parse"}.get(mode, mode), "config": cfg, "n": len(d),
                     "old_median": float(d.pin_s_old.median()), "new_median": float(d.pin_s_new.median()),
                     "ratio_median": float(r.median()), "ratio_lo": float(r.min()), "ratio_hi": float(r.max()),
                     "spearman": float(d.pin_s_new.rank().corr(d.pin_s_old.rank()))})
        if mode == "splitter":
            rn = d.native_s_new / d.native_s_old
            rows.append({"what": "Native run", "config": cfg, "n": len(d), "old_median": float(d.native_s_old.median()),
                         "new_median": float(d.native_s_new.median()), "ratio_median": float(rn.median()),
                         "ratio_lo": float(rn.min()), "ratio_hi": float(rn.max()),
                         "spearman": float(d.native_s_new.rank().corr(d.native_s_old.rank()))})
    enu = S[(S.part == "pb_interp") & (S.step == "interpreter (enumerated)")].rename(columns={"workload": "w"})
    oi = old[old["mode"] == "interpreter"].rename(columns={"workload": "w"})
    e = enu.merge(oi, left_on=["w", "config"], right_on=["w", "config"])
    for cfg, d in e.groupby("config"):
        r = d.seconds / d.pin_s
        rows.append({"what": "C interpreter, enumerated (same method)", "config": cfg, "n": len(d),
                     "old_median": float(d.pin_s.median()), "new_median": float(d.seconds.median()),
                     "ratio_median": float(r.median()), "ratio_lo": float(r.min()), "ratio_hi": float(r.max()),
                     "spearman": float(d.seconds.rank().corr(d.pin_s.rank()))})
    rec = recorded(ROOT / "data/gap-bytes/data/static_auto", "*-16.npz")
    ser = S[(S.part == "cpp")].set_index("workload").seconds
    if len(ser):
        o = pd.Series({k: rec.get(f"{k}-16", np.nan) for k in ser.index})
        r = ser / o
        rows.append({"what": "C++ interpreter, GAP scale 16 (recorded in parallel, now alone)", "config": "16",
                     "n": len(ser), "old_median": float(o.median()), "new_median": float(ser.median()),
                     "ratio_median": float(r.median()), "ratio_lo": float(r.min()), "ratio_hi": float(r.max()),
                     "spearman": float(ser.rank().corr(o.rank()))})
    v = pd.DataFrame(rows)
    v.to_csv(D / "costs_verify.csv", index=False)
    return v


def fmt_s(x):
    if pd.isna(x):
        return "--"
    if x < 0.01:
        return f"{x * 1000:.1f}\,ms"
    if x < 1:
        return f"{x * 1000:.0f}\,ms"
    if x < 100:
        return f"{x:.1f}"
    if x < 7200:
        return f"{x:.0f}"
    return f"{x / 3600:.1f}\,h"


def fmt_x(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}" if x < 10 else f"{x:.0f}"


def write_tex(out, S, v):
    END = r" \\"
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Method & Runs of the program & s & $\times$native & $\times$full trace & $\times$sampled run" + END,
             r"\midrule"]
    for g, d in out.groupby("group", sort=False):
        lines.append(r"\multicolumn{6}{l}{\textit{" + g + "}}" + END)
        for _, r in d.iterrows():
            lines.append(r"\quad " + f"{r.method} & {r.runs} & {fmt_s(r.seconds)} & {fmt_x(r.x_native)} & "
                         f"{fmt_x(r.x_trace)} & {fmt_x(r.x_sampled)}" + END)
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "costs_all.tex").write_text("\n".join(lines) + "\n")
    st = S[S.part == "pb_analysis"].groupby("step").agg(med=("seconds", "median"), mx=("seconds", "max"), n=("seconds", "size"))
    order = ["runtime baseline (NNLS on 26)", "moments at 13 intervals (static alpha)", "static alpha for the sampled bins",
             "descriptor z-hat", "known programs' measured z", "distances and RBF weights", "apply the 26 borrowed models",
             "nearest model (pick, apply)", "RBF mixture (combine)", "pooled regression with static alpha (fit and apply)",
             "own model: fit on 6 configs and apply", "fit the 27 per-program models"]
    pi = S[(S.part == "pb_interp") & S.step.isin(["clang parse", "AST node counts", "access-idiom features"])] \
        .groupby("step").agg(med=("seconds", "median"), mx=("seconds", "max"), n=("seconds", "size"))
    sl = [r"\begin{tabular}{lrrr}", r"\toprule", "Step & median (ms) & max (ms) & runs" + END, r"\midrule"]
    for k in order + list(pi.index):
        r = st.loc[k] if k in st.index else pi.loc[k] if k in pi.index else None
        if r is not None:
            name = k.replace("z-hat", r"$\hat z$").replace("alpha", r"$\alpha$").replace("measured z", "measured $z$")
            sl.append(f"{name} & {r.med * 1000:.1f} & {r.mx * 1000:.1f} & {int(r.n)}" + END)
    sl += [r"\bottomrule", r"\end{tabular}"]
    (T / "costs_steps.tex").write_text("\n".join(sl) + "\n")
    vl = [r"\begin{tabular}{llrrrrr}", r"\toprule",
          "Measured & Config & Before (s) & Now (s) & Now/before & range & Spearman" + END, r"\midrule"]
    for _, r in v.iterrows():
        vl.append(f"{r.what} & {r.config} & {fmt_s(r.old_median)} & {fmt_s(r.new_median)} & {r.ratio_median:.2f} & "
                  f"{r.ratio_lo:.2f}--{r.ratio_hi:.2f} & {r.spearman:.2f}" + END)
    vl += [r"\bottomrule", r"\end{tabular}"]
    (T / "costs_verify.tex").write_text("\n".join(vl) + "\n")


def wide_table(out):
    """One row per program set and size: the references (native, Pin) and the static methods."""
    END = r" \\"
    static_methods = ["Static footprint: C interpreter, symbolic counting", "C interpreter, every iteration enumerated",
                      "Footprint over time: C interpreter, timeline mode", "Hand skeleton and its spectrum",
                      "C++ interpreter (one at a time)", "C++ interpreter (recorded, 4--16 jobs at a time)",
                      "C++ interpreter (recorded)", "C interpreter"]
    short = {"Static footprint: C interpreter, symbolic counting": "C interpreter, symbolic",
             "C interpreter, every iteration enumerated": "C interpreter, enumerated",
             "Footprint over time: C interpreter, timeline mode": "C interpreter, timeline",
             "Hand skeleton and its spectrum": "hand skeleton", "C++ interpreter (one at a time)": "C++ interpreter, alone",
             "C++ interpreter (recorded, 4--16 jobs at a time)": "C++ interpreter, 4--16 at a time",
             "C++ interpreter (recorded)": "C++ interpreter", "C interpreter": "C interpreter"}
    lines = [r"\begin{tabular}{lrrrrlrrr}", r"\toprule",
             r"Programs (size) & native & \multicolumn{3}{c}{Pin ($\times$native)} & Static method & s & $\times$native & $\times$trace" + END,
             r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
             r" & s & instr. & sampled & full trace & & & &" + END, r"\midrule"]
    keep = ["PolyBench (27 kernels, MEDIUM)", "GAP (12 workloads, scale 16)", "GAP (12 workloads, scale 18)",
            r"C program csr\_pr (scale 16)", r"C program csr\_bfs (scale 14)", "HPCCG (28)", "LULESH (20)",
            "NPB IS (S)", "NPB IS (W)", "NPB CG (S)", "NPB CG (W)", "NPB MG (S)", "NPB MG (W)", "NPB MG (A)"]
    for g in keep:
        d = out[out.group == g].set_index("method")
        if d.empty:
            continue
        get = lambda m, c: d.loc[m, c] if m in d.index else np.nan
        nat = get("Native run", "seconds")
        ins, smp = get("Pin, instrumentation only", "x_native"), get("Sampled run", "x_native")
        if pd.isna(smp):
            smp = get("Sampled run (\\memprint{}'s deployed run)", "x_native")
        trc = get("Full trace", "x_native")
        if pd.isna(trc):
            trc = get("Full trace (Pin splitter; exact footprint)", "x_native")
        trc_s = "out of memory" if g == "NPB MG (A)" else fmt_x(trc)
        ms = [m for m in static_methods if m in d.index]
        label = g.replace(" (12 workloads, ", ", 12 workloads (").replace("C program ", "")
        for i, m in enumerate(ms):
            head = f"{label} & {fmt_s(nat)} & {fmt_x(ins)} & {fmt_x(smp)} & {trc_s}" if i == 0 else " & & & &"
            lines.append(f"{head} & {short[m]} & {fmt_s(d.loc[m, 'seconds'])} & {fmt_x(d.loc[m, 'x_native'])} & "
                         f"{fmt_x(d.loc[m, 'x_trace'])}" + END)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "costs_wide.tex").write_text("\n".join(lines) + "\n")


def gap_cost_table():
    """tables/cost.tex: the C++ interpreter's recorded seconds per GAP workload and scale (data/gap-bytes static_auto)."""
    rec = recorded(ROOT / "data/gap-bytes/data/static_auto", "*.npz")
    END = r" \\"
    cols = [(k, g) for k in ("pr", "bfs", "cc", "bc", "tc", "sssp") for g in ("", "_kron")]
    lines = [r"\begin{tabular}{r" + "r" * len(cols) + "}", r"\toprule",
             "Scale & " + " & ".join(f"{k} {'K' if g else 'u'}" for k, g in cols) + END, r"\midrule"]
    for sc in range(10, 19):
        vals = [rec.get(f"gap_{k}{g}-{sc}", np.nan) for k, g in cols]
        lines.append(f"{sc} & " + " & ".join("--" if pd.isna(v) else f"{v:.0f}" for v in vals) + END)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "cost.tex").write_text("\n".join(lines) + "\n")
    return pd.Series(rec)


def numbers(out, S):
    q = lambda part, step, cfg: S[(S.part == part) & (S.step == step) & (S.config == cfg)].set_index("workload").seconds
    sym, tl, enu = (q("pb_interp", s, "MEDIUM") for s in ("interpreter (symbolic)", "interpreter (timeline)",
                                                         "interpreter (enumerated)"))
    nums = {"tlCostX": float((tl / sym).median()), "enuCostX": float((enu / sym).median()),
            "symMedianS": float(sym.median()), "tlMedianS": float(tl.median())}
    st = S[S.part == "pb_analysis"].groupby("step").seconds.median()
    nums["baselineFitMs"] = float(st.get("runtime baseline (NNLS on 26)", np.nan) * 1000)
    nums["momentsMs"] = float(st.get("moments at 13 intervals (static alpha)", np.nan) * 1000)
    nums["pooledMs"] = float(st.get("pooled regression with static alpha (fit and apply)", np.nan) * 1000)
    sk = S[S.part == "gap_skeleton"]
    if len(sk):
        nums["skelMedianS"] = float(sk.seconds.median())
        nums["skelMaxS"] = float(sk.seconds.max())
    cp = S[S.part == "cpp"]
    if len(cp):
        nums["cppSixteenMedianS"] = float(cp.seconds.median())
    nums["loadMedian"] = float(S.load.median())
    g = pd.Series(recorded(ROOT / "data/gap-bytes/data/static_auto", "*.npz"))
    if len(g):
        nums["GapPrKronEighteen"] = float(g.get("gap_pr_kron-18", np.nan))
        nums["GapTotalHours"] = float(g.sum() / 3600)
    text = "".join(r"\newcommand{\numc" + k + "}{" + (f"{v:,.0f}" if v >= 10 else f"{v:.2f}") + "}\n"
                   for k, v in nums.items() if pd.notna(v))
    (T / "numbers_costs.tex").write_text(text)
    return nums


if __name__ == "__main__":
    out, S = main()
    new, complete = measured_pb_timings(S)
    v = verify(S, new)
    print(v.round(3).to_string())
    write_tex(out, S, v)
    wide_table(out)
    g = gap_cost_table()
    print("GAP interpreter: total", round(g.sum() / 3600, 2), "h; max", g.idxmax(), round(g.max()))
    print(numbers(out, S))
    print("pb_timings_measured.csv written" if complete else "PolyBench measurements incomplete")
