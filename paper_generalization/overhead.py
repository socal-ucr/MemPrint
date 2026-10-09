"""Cost of every method of the leave-one-out table, from measured run times.

python paper_generalization/overhead.py   (reads data/pb_timings.csv, writes data/overhead_*.csv)

data/pb_timings.csv holds, per PolyBench kernel and config, the time of
    splitter     the full Pin trace (MemPrint's ground truth), all seven configs, with the native time
    sampler      one sampled Pin run with run.sh's defaults (-i 1000 -s 100 -r 10), SMALL2 and MEDIUM
    instr        Pin with instrumentation only (a sampler that never samples), SMALL2 and MEDIUM
    interpreter  the static interpreter (bytes touched), SMALL2 and MEDIUM, one run at a time
    clang parse  parsing the source with libclang (the AST-similarity baseline), SMALL2 and MEDIUM
Splitter and native times come from the original traces; sampler and instrumentation runs were measured
with four kernels in parallel on a 32-core node.

Per held-out kernel C and split (EXTRA: MEDIUM, INTER: SMALL2), the work each method needs for C:
    static-fp          the interpreter at the test config; no run of C
    static-alpha, pooled-static, nn-static, mix-static
                       the interpreter + one sampled run (their alpha scales that run's bin footprint)
    nn-ast             a clang parse + one sampled run
    mean               one sampled run
    own                full traces of C's six other configs (training) + one sampled run
    nn-measured, mix-measured
                       a full trace of C at the test config (for its z) + one sampled run; that trace
                       already gives the exact footprint, so these are upper bounds, not methods
    oracle             a full trace at the test config (choosing needs the truth)
The borrowing methods, the pooled regression and the runtime baseline also need full traces of other
kernels, once (reported separately).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PB = HERE.parent / "data" / "polybench-bytes" / "data"
CONFIGS = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]
SPLITS = {"EXTRA": "MEDIUM", "INTER": "SMALL2"}
ORDER = ["native", "instrumentation only", "full trace", "static-fp", "static-alpha", "pooled-static", "nn-static",
         "mix-static", "nn-ast", "mean", "own", "nn-measured", "mix-measured", "oracle"]


def main():
    t = pd.read_csv(HERE / "data" / "pb_timings.csv")
    T = t.set_index(["workload", "config", "mode"])
    rows = []
    for k in sorted(t.workload.unique()):
        for split, c in SPLITS.items():
            native = float(T.loc[(k, c, "splitter"), "native_s"])
            full = float(T.loc[(k, c, "splitter"), "pin_s"])
            samp = float(T.loc[(k, c, "sampler"), "pin_s"])
            interp = float(T.loc[(k, c, "interpreter"), "pin_s"])
            parse = float(T.loc[(k, c, "clang parse"), "pin_s"])
            train = float(sum(T.loc[(k, o, "splitter"), "pin_s"] for o in CONFIGS if o != c))
            cost = {"native": native, "instrumentation only": float(T.loc[(k, c, "instr"), "pin_s"]),
                    "full trace": full, "static-fp": interp, "static-alpha": interp + samp,
                    "pooled-static": interp + samp, "nn-static": interp + samp, "mix-static": interp + samp,
                    "nn-ast": parse + samp, "mean": samp, "own": train + samp, "nn-measured": full + samp,
                    "mix-measured": full + samp, "oracle": full}
            for m, v in cost.items():
                rows.append({"workload": k, "split": split, "config": c, "method": m, "seconds": v,
                             "x_native": v / native, "x_full_trace": v / full})
    per = pd.DataFrame(rows)
    per.to_csv(HERE / "data" / "overhead_per_kernel.csv", index=False)
    err = pd.read_csv(PB / "lowo_errors.csv").groupby(["method", "split"]).mape.median()
    out = []
    for m in ORDER:
        for split in SPLITS:
            d = per[(per.method == m) & (per.split == split)]
            out.append({"method": m, "split": split, "median_s": d.seconds.median(),
                        "median_x_native": d.x_native.median(), "median_x_full_trace": d.x_full_trace.median(),
                        "median_mape": err.get((m, split), np.nan)})
    out = pd.DataFrame(out)
    out.to_csv(HERE / "data" / "overhead_methods.csv", index=False)
    sp = t[t["mode"] == "splitter"]
    q = per.pivot_table(index=["workload", "split"], columns="method", values="seconds")
    facts = {"one_time_s": float(sp.pin_s.sum()), "kernels": int(t.workload.nunique()),
             "interp_slower_than_sampler_extra": int((q.xs("EXTRA", level="split")["static-fp"] >
                                                     q.xs("EXTRA", level="split")["mean"]).sum()),
             "interp_slower_than_sampler_inter": int((q.xs("INTER", level="split")["static-fp"] >
                                                     q.xs("INTER", level="split")["mean"]).sum()),
             "interp_x_native_max": float(per[per.method == "static-fp"].x_native.max()),
             "smallest_full_trace_s": float(sp[sp.config == "MINI"].pin_s.min())}
    pd.Series(facts).to_csv(HERE / "data" / "overhead_facts.csv", header=["value"])
    print(out.round(3).to_string(index=False))
    print(facts)


if __name__ == "__main__":
    main()
