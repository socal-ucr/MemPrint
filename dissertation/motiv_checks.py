"""The assumptions behind Chapter 2's motivating gain (2.Motivation.tex, sec:mp-simulator; PROTOCOL.md "Motivation
checks"), on the motivation's own setting: the 2026-10-13-sched-llm corpus (25 programs, largest inputs), the
synthetic stream (3,000 jobs, lognormal requests with median 100 GB and sigma 0.8, u from LANL CM5), offered load
0.8, seeds 0 to 2, both truths (bytes touched and RSS). Results -> $R/results/<part>.csv.

  e1      memprint-peak (true peak x (1 + e), e one of MemPrint's signed L2O extrapolation errors on the 35-workload
          set, dissertation/mp_errors.py) for the bootstrap, disjoint and splitter bins, every margin policy
  e1cap   memprint-peak (bootstrap) over the memory sweep (48 to 512 GB per node)
  e2      the request a fixed factor (1.5, 2, 3) above the peak instead of CM5's u, every oracle of tab:mp-load08
  e2mc    the MetaCentrum2 stream (its requests and arrivals; u from CM5), the same oracles
  e3      fixed parts kept unscaled (real.fixed_parts), the same oracles
  e3cap   the same over the memory sweep
  report  tables (dissertation/tables/motiv_*.tex) and the numbers the text quotes

    OMP_NUM_THREADS=1 PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/motiv_checks.py <part ...>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "paper_sched"))
from memprint.sched import run  # noqa: E402
from memprint.sched.model import Cluster  # noqa: E402

D = os.path.expanduser("~/memprint-data/2026-10-13-sched-llm")
R = os.path.expanduser("~/memprint-data/2026-11-18-motivchecks")
# the 37-workload errors with the high-bit keep test (hibits_score.py); the old tool's E1 results are in results/oldtool
MP = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2/mp_errors_hb.csv")
PROCS = int(os.environ.get("PROCS", 28))
TAB = os.path.join(HERE, "tables")
# the oracles of tab:mp-load08 (the motivation's grid_08), and MemPrint's sampled peak
LOAD08 = ["exact", "exact-coarse", "exact-peak", "zacarias", "zacarias-past", "reactive", "rss-peak", "dhat", "ksplus",
          "ml-peak", "request", "massif-i", "pond", "dhat-curve", "dhat-hold", "massif", "massif-pages-i", "massif-pages",
          "gperf", "heaptrack"]
SWEEP = ["dhat", "dhat-curve", "exact", "exact-coarse", "exact-peak", "gperf", "heaptrack", "massif", "massif-i",
         "ml-peak", "reactive", "request", "rss-peak", "pond", "zacarias", "ksplus"]
SIZES = (48, 64, 96, 128, 160, 192, 256, 320, 384, 512)
COMMON = dict(loads=(0.8,), seeds=(0, 1, 2), n=3000, procs=PROCS, scenarios=("bytes", "rss"))
MPO = dict(mp=MP, mp_sampler="boot")


def go(name, names, margins=("cal",), kind="synthetic", cluster=None, **opts):
    out = f"{R}/results/{name}.csv"
    if os.path.exists(out):
        print("exists", out)
        return
    run.run(D + "/corpus", D + "/traces", out, names=names, margins=margins, kind=kind, cluster=cluster or Cluster(),
            opts=opts, **COMMON)
    print("done", name, flush=True)


def e1():
    for s in ("boot", "disj", "split"):
        go(f"e1_{s}", ["memprint-peak"], margins=("cal", "cal100", 0.1, 0.3), mp=MP, mp_sampler=s)


OWN = os.path.expanduser("~/memprint-data/2026-11-23-corpus-e1/mp_errors_corpus.csv")   # dissertation/corpus_e1.py


def e1own():
    """Full E1: each job's error drawn from its own program's runs (corpus_e1.py), the 37-workload bootstrap errors for
    a program without them (llama2c, hpccg until rerun); own_floor with alpha clipped at 1."""
    go("e1own_boot", ["memprint-peak"], margins=("cal", "cal100", 0.1, 0.3), mp=MP, mp_sampler="boot", mp_own=OWN)
    go("e1own_floor", ["memprint-peak"], margins=("cal", "cal100", 0.1, 0.3), mp=MP, mp_sampler="boot",
       mp_own=OWN.replace(".csv", "_floor.csv"))


def e1cap():
    for gb in SIZES:
        go(f"e1cap_{gb}", ["memprint-peak"], cluster=Cluster(local=gb), **MPO)


def e1cap_disj():
    for gb in SIZES:
        go(f"e1capdisj_{gb}", ["memprint-peak"], cluster=Cluster(local=gb), mp=MP, mp_sampler="disj")


def e2():
    for f in (1.5, 2.0, 3.0):
        go(f"e2_x{f}", LOAD08 + ["memprint-peak"], overestimate=f, **MPO)


def e2mc():
    go("e2_metacentrum", LOAD08 + ["memprint-peak"], kind="trace", **MPO)


def e3():
    go("e3", LOAD08 + ["memprint-peak"], fixed=True, **MPO)


def e3cap():
    for gb in SIZES:
        go(f"e3cap_{gb}", SWEEP + ["memprint-peak"], cluster=Cluster(local=gb), fixed=True, **MPO)


def _load(pattern, root=R):
    files = sorted(glob.glob(f"{root}/results/{pattern}"))
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "error" in g:
        bad = int(g.error.notna().sum())
        if bad:
            print(pattern, bad, "failed runs left out")
        g = g[g.error.isna()]
    g["margin"] = g.margin.astype(str)
    if "restarts" in g:
        g["kills"] = g.kills + g.restarts.fillna(0)
    return g


def _cal(g):
    from make_figures import select
    return select(g, "cal")


def _summ(g):
    return g.groupby(["scenario", "oracle"]).agg(resp=("steady_response", "mean"), kills=("kills", "mean"),
                                                 res=("steady_reserved", "mean"), used=("steady_resident", "mean"))


def _share(s, sc, o):
    r, p = s.loc[(sc, "request"), "resp"], s.loc[(sc, "exact-peak"), "resp"]
    return (r - s.loc[(sc, o), "resp"]) / (r - p)


def _tb(cap, targets):
    from make_figures import capacity_needed
    n = capacity_needed(cap, targets)
    n["tb"] = (32 * n.local_gb + 1024) / 1024   # TB of 1,024 GB, as the motivation quotes
    return n.set_index(["scenario", "oracle"]).tb


def report():
    from make_figures import capacity_targets
    from make_tables import run_label
    pd.set_option("display.width", 220)
    base = _summ(_cal(_load("grid_all*.csv", D)[lambda x: x.load == 0.8]))
    cap0 = _load("capacity_*.csv", D)
    cap0 = cap0[cap0.load == 0.8]
    targets = capacity_targets(cap0)
    tb0 = _tb(cap0, targets)
    lab = lambda o: "MemPrint's sampled peak" if o == "memprint-peak" else run_label(o)
    # E1
    rows = []
    e1 = _load("e1_*.csv")
    e1cap = _load("e1cap_*.csv")
    tb1 = _tb(pd.concat([cap0, e1cap]), targets)
    for s, name in (("boot", "Bootstrap sampler"), ("disj", "Disjoint sampler"), ("split", "Splitter bins")):
        g = e1[e1.opt_mp_sampler == s]
        for m, mname in (("cal", "calibrated 90\\%"), ("cal100", "calibrated 100\\%"), ("0.1", "10\\%"), ("0.3", "30\\%")):
            h = g[g.margin == m].groupby("scenario").agg(resp=("steady_response", "mean"), kills=("kills", "mean"))
            cells = [name if m == "cal" else "", mname]
            for sc in ("bytes", "rss"):
                r, p = base.loc[(sc, "request"), "resp"], base.loc[(sc, "exact-peak"), "resp"]
                cells += [f"{h.loc[sc, 'resp']:,.0f}".replace(",", "{,}"), f"{h.loc[sc, 'kills']:,.0f}".replace(",", "{,}"),
                          f"{(r - h.loc[sc, 'resp']) / (r - p) * 100:.0f}"]
            rows.append(" & ".join(cells) + " \\\\")
            print("E1", s, m, h.round(1).to_dict())
    with open(f"{TAB}/motiv_e1.tex", "w") as f:
        f.write("\n".join(rows) + "\n")
    for sc in ("bytes", "rss"):
        r, p, mp = tb0[(sc, "request")], tb0[(sc, "exact-peak")], tb1.get((sc, "memprint-peak"))
        print(f"E1 TB {sc}: memprint-peak {mp} TB, exact peak {p}, request {r}, DHAT {tb0[(sc, 'dhat')]}; share of the bound "
              f"{(r - mp) / (r - p) * 100 if mp == mp else float('nan'):.0f}%; DHAT {(r - tb0[(sc, 'dhat')]) / (r - p) * 100:.0f}%")
    # E2
    sets = [("CM5 $u$ (as published)", base)]
    for x in (1.5, 2.0, 3.0):
        sets.append((f"{x:g}$\\times$ the peak", _summ(_cal(_load(f"e2_x{x}.csv")))))
    mc = glob.glob(f"{R}/results/e2_metacentrum.csv")
    if mc:
        sets.append(("MetaCentrum2, CM5 $u$", _summ(_cal(_load("e2_metacentrum.csv")))))
    e1b = _summ(_cal(e1[e1.opt_mp_sampler == "boot"]))
    sets[0] = (sets[0][0], pd.concat([base, e1b]))
    oracles = [o for o in LOAD08 + ["memprint-peak"] if o not in ("request", "exact-peak")]
    rows = []
    for sc in ("bytes", "rss"):
        head = [f"Request / exact peak, {'bytes touched' if sc == 'bytes' else 'RSS'}"]
        for _, s in sets:
            head.append(f"{s.loc[(sc, 'request'), 'resp'] / s.loc[(sc, 'exact-peak'), 'resp']:.1f}")
        rows.append(" & ".join(head) + " \\\\")
    rows.append("\\midrule")
    for o in oracles:
        cells = [lab(o)]
        for _, s in sets:
            cells.append(f"{_share(s, 'bytes', o) * 100:.0f}" if ("bytes", o) in s.index else "--")
        rows.append(" & ".join(cells) + " \\\\")
    with open(f"{TAB}/motiv_e2.tex", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("E2 columns:", [n for n, _ in sets])
    for sc in ("bytes", "rss"):
        for n, s in sets:
            print(f"E2 {sc} {n}: request {s.loc[(sc, 'request'), 'resp']:.0f} exact-peak {s.loc[(sc, 'exact-peak'), 'resp']:.0f} "
                  f"exact {s.loc[(sc, 'exact'), 'resp']:.0f} ratio {s.loc[(sc, 'request'), 'resp'] / s.loc[(sc, 'exact-peak'), 'resp']:.2f}; "
                  f"shares: " + ", ".join(f"{o} {_share(s, sc, o) * 100:.0f}" for o in ("dhat", "rss-peak", "massif-i", "memprint-peak", "reactive", "zacarias")
                                          if (sc, o) in s.index))
    # E3
    e3 = _summ(_cal(_load("e3.csv")))
    order = base.loc["bytes"].resp.sort_values().index
    rows = []
    for o in [o for o in order if ("bytes", o) in e3.index] + (["memprint-peak"] if ("bytes", "memprint-peak") in e3.index else []):
        cells = [lab(o)]
        for sc in ("bytes", "rss"):
            b = pd.concat([base, e1b]).loc[(sc, o)] if (sc, o) in pd.concat([base, e1b]).index else None
            cells += [f"{b.resp:,.0f}".replace(",", "{,}") if b is not None else "--",
                      f"{e3.loc[(sc, o), 'resp']:,.0f}".replace(",", "{,}"), f"{e3.loc[(sc, o), 'kills']:,.0f}".replace(",", "{,}")]
        rows.append(" & ".join(cells) + " \\\\")
    with open(f"{TAB}/motiv_e3.tex", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("E3 order (bytes, proportional):", list(order))
    print("E3 order (bytes, fixed):", list(e3.loc["bytes"].resp.sort_values().index))
    setup0 = run.Setup(f"{D}/corpus", f"{D}/traces", 10, Cluster(), "synthetic")
    setup3 = run.Setup(f"{D}/corpus", f"{D}/traces", 10, Cluster(), "synthetic", opts=dict(fixed=True))
    rows = []
    for o in ["massif-i", "massif-pages-i", "massif", "massif-pages", "heaptrack", "dhat", "dhat-curve", "dhat-hold", "gperf", "rss-peak"]:
        cells = [run_label(o)]
        for sc in ("bytes", "rss"):
            for st in (setup0, setup3):
                v = np.array(list(st.required(sc, o).values()))
                cells += [f"{np.median(v) * 100:.0f}", f"{np.quantile(v, 0.9) * 100:.0f}"]
        rows.append(" & ".join(cells) + " \\\\")
    with open(f"{TAB}/motiv_e3_margins.tex", "w") as f:
        f.write("\n".join(rows) + "\n")
    caps = glob.glob(f"{R}/results/e3cap_*.csv")
    if caps:
        c3 = _load("e3cap_*.csv")
        tb3 = _tb(c3, capacity_targets(c3))
        tbp = _tb(pd.concat([cap0, e1cap]), targets)
        rows = []
        for o in SWEEP + ["memprint-peak"]:
            cells = [lab(o)]
            for sc in ("bytes", "rss"):
                for t in (tbp, tb3):
                    v = t.get((sc, o), np.nan)
                    cells.append("--" if v != v else f"{v:.1f}")
            rows.append(" & ".join(cells) + " \\\\")
        with open(f"{TAB}/motiv_e3_tb.tex", "w") as f:
            f.write("\n".join(rows) + "\n")
        print("E3 TB targets", capacity_targets(c3), "published", targets)
        print(pd.DataFrame({"prop": tbp, "fixed": tb3}).to_string())


if __name__ == "__main__":
    os.makedirs(f"{R}/results", exist_ok=True)
    for part in sys.argv[1:]:
        globals()[part]()
