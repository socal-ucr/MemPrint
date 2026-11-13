"""Tables of the scheduling results under the pages-in-use truth (sched_pages.py run) for the dissertation's
resident-memory chapter, in the layout of paper_sched/make_tables.py, with the columns "Truth = pages in use" and
"RSS".

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/sched_pages_tables.py   ->  paper_rss/tables/sched/

sched_policies.tex   every oracle at load 0.8: response, bounded slowdown, kills, memory reserved and used
sched_loads.tex      mean response time (min) of every oracle at every load
sched_llm.tex        the LLM stream at load 0.8, as sched_policies
sched_capacity.tex   total memory (TB) each oracle needs for the service target, at every load
sched_sens_{beta,pool,interval}.tex
sched_lisa_hybrid.tex
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "paper_sched"))
import make_figures as MF  # noqa: E402
import sensitivity as SE  # noqa: E402
from sched_pages import DYN, MODEL1, P, SENS  # noqa: E402

OUT = os.path.join(HERE, "tables", "sched")
SC = ("pagesnat", "rss")
TITLES = ("Truth = pages in use", "RSS")
MODEL_LABELS = {"rss-peak-rep": "RSS peak, repeat run", "rss-curve-rep": "RSS curve, repeat run",
                "memprint-W5nat": r"\memprint{} W5, native clock", "model1-rep": r"Data $\hat f_1$, repeat run",
                "model2-rep": r"Charged $\hat f_2$, repeat run", "rss-peak-x": "RSS peak, smaller inputs",
                "rss-curve-x": "RSS curve, smaller inputs", "memprint-W5-x": r"\memprint{} W5, smaller inputs",
                "memprint-W5nat-x": r"\memprint{} W5, native clock, smaller inputs",
                "model1-x": r"Data $\hat f_1$, smaller inputs", "model2-x": r"Charged $\hat f_2$, smaller inputs"}


def label(o):
    return MODEL_LABELS.get(o) or MF.style(o)[0].replace("%", r"\%")


def fmt(x, d=0):
    return "--" if pd.isna(x) else f"{x:,.{d}f}".replace(",", "{,}")


def load(pattern):
    """The rows of $P/results/<pattern> (failed runs left out, failed resizes counted as kills). A file's rows for the
    model's oracles (sched_pages.MODEL1) are replaced by those of its <name>_model.csv, run on the corpus with the LLM
    engines' model curves (sched_pages.py run model); a "file" column keeps each row's source without "_model"."""
    import glob
    files = sorted(glob.glob(f"{P}/results/{pattern}"))
    model = {f[:-len("_model.csv")] + ".csv" for f in files if f.endswith("_model.csv")}
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if f in model:
            d = d[~d.oracle.isin(MODEL1)]
        frames.append(d.assign(file=os.path.basename(f).replace("_model.csv", ".csv")))
    g = pd.concat(frames, ignore_index=True)
    g = g[g.error.isna()] if "error" in g else g
    g["margin"] = g.margin.astype(str)
    if "restarts" in g:
        g["kills"] = g.kills + g.restarts.fillna(0)
    return g


def grid(pattern, sub=""):
    """The calibrated rows of $P/results/<sub><pattern> (and its _model file)."""
    p = sub + pattern
    return MF.select(load(p if "*" in p else p.replace(".csv", "*.csv")), "cal")


def head(cols, sub, n):
    return [r"\begin{tabular}{l" + "r" * (n * 2) + "}", r"\toprule",
            " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{t}}}" for t in TITLES) + r" \\",
            f"\\cmidrule(lr){{2-{1 + n}}} \\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}", sub + " & " + " & ".join(cols * 2) + r" \\",
            r"\midrule"]


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    print("wrote", name)


def policies(g, name, load=0.8):
    g = g[g.load == load]
    order = g[g.scenario == SC[0]].groupby("oracle").steady_response.mean().sort_values().index
    lines = head(["Resp.\\ (min)", "BSLD", "Kills", "Res.\\ \\%", "Used \\%"], "Oracle", 5)
    for o in order:
        cells = [label(o)]
        for sc in SC:
            d = g[(g.scenario == sc) & (g.oracle == o)]
            cells += [fmt(d.steady_response.mean()), fmt(d.steady_bsld.mean(), 1), fmt(d.kills.mean()),
                      fmt(d.steady_reserved.mean() * 100), fmt(d.steady_resident.mean() * 100)]
        lines.append(" & ".join(cells) + r" \\")
    write(name, lines)


def loads(g, name):
    ls = sorted(g.load.unique())
    m = g.groupby(["scenario", "oracle", "load"]).steady_response.mean()
    order = m.loc[SC[0]].xs(0.8, level="load").sort_values().index
    lines = head([f"{x:g}" for x in ls], "Load", len(ls))
    for o in order:
        lines.append(" & ".join([label(o)] + [fmt(m.get((sc, o, x))) for sc in SC for x in ls]) + r" \\")
    write(name, lines)


def capacity():
    c = load("capacity_l*.csv")
    need = MF.capacity_needed(c)
    need["tb"] = (need.local_gb * 32 + 1024) / 1024
    ls = sorted(need.load.unique())
    t = need.set_index(["oracle", "scenario", "load"]).tb
    first = need[(need.scenario == SC[0]) & (need.load == 0.8)].sort_values("tb", na_position="last")
    lines = head([f"{x:g}" for x in ls], "Load", len(ls))
    for o in first.oracle:
        lines.append(" & ".join([label(o)] + [fmt(t.get((o, sc, x)), 1) for sc in SC for x in ls]) + r" \\")
    write("sched_capacity", lines)
    need.to_csv(f"{P}/capacity_needed.csv", index=False)


def sensitivity():
    import glob
    import re
    rows = []
    sens = load("sens/sched_*.csv")
    for f, d in sens.groupby("file"):
        k, v = re.match(r"sched_(\w+?)_([\d.]+)\.csv", f).groups()
        for (sc, o), h in MF.select(d, "cal").groupby(["scenario", "oracle"]):
            rows.append(dict(param=k, value=float(v), scenario=sc, oracle=o, resp=h.steady_response.mean()))
    base = grid("grid_all.csv")
    for (sc, o), h in base[base.load == 0.8].groupby(["scenario", "oracle"]):
        for k, v in SE.DEFAULT.items():
            rows.append(dict(param=k, value=v, scenario=sc, oracle=o, resp=h.steady_response.mean()))
    s = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    for param, names in (("beta", SENS), ("pool", SENS), ("interval", DYN)):
        SE.table(s, param, names, label, "oracle", SE.n0, f"{OUT}/sched_sens_{param}.tex", groups=SC, group_titles=TITLES)
        print("wrote sched_sens_" + param)


def lisa():
    g = grid("lisa_hybrid_32.csv", "lisa/")
    g = g[g.load == 0.8]
    order = g[g.scenario == SC[0]].groupby("oracle").steady_response.mean().sort_values().index
    lines = head(["Jobs/h", "Resp.\\ (min)", "Fails"], "Oracle", 3)
    for o in order:
        cells = [label(o)]
        for sc in SC:
            d = g[(g.scenario == sc) & (g.oracle == o)]
            cells += [fmt(d.throughput.mean(), 1), fmt(d.steady_response.mean()), fmt(d.kills.mean())]
        lines.append(" & ".join(cells) + r" \\")
    write("sched_lisa_hybrid", lines)


def summary():
    """The numbers the text quotes: per stream and truth at load 0.8, the response of the exact curve, the exact peak
    and the request, and the exact peak over the exact curve -> $P/summary.csv."""
    rows = []
    for stream in ("grid_all", "grid_llm"):
        if not os.path.exists(f"{P}/results/{stream}.csv"):
            continue
        g = grid(f"{stream}.csv")
        m = g[g.load == 0.8].groupby(["scenario", "oracle"]).agg(resp=("steady_response", "mean"), kills=("kills", "mean"))
        for sc in SC:
            r = m.loc[sc]
            rows.append(dict(stream=stream, scenario=sc, **{f"{o}_resp": r.resp.get(o) for o in r.index},
                             **{f"{o}_kills": r.kills.get(o) for o in r.index},
                             peak_over_curve=r.resp["exact-peak"] / r.resp["exact"]))
    d = pd.DataFrame(rows)
    d.to_csv(f"{P}/summary.csv", index=False)
    cols = ["exact_resp", "exact-peak_resp", "request_resp", "peak_over_curve"]
    print(d[["stream", "scenario"] + cols].round(2).to_string(index=False))


def model_grid():
    """paper_rss/compute.py's grid tables (tab_grid: the model's rows and its comparisons, response at loads 0.6, 0.8
    and 1.0, and throughput, kills and wasted core-hours at 0.8, means of the 3 seeds) on this corpus ->
    sched_model_{pagesnat,rss}.tex."""
    import compute as CP
    g = load("grid_all*.csv")
    pick = g.assign(rank=g.margin.map({"cal": 0, "0.1": 1, "0.0": 2}).fillna(3))
    pick = pick[pick["rank"] == pick.groupby(["scenario", "oracle"])["rank"].transform("min")]
    for sc in SC:
        m = pick[pick.scenario == sc].groupby(["oracle", "load"])[["response", "throughput", "kills", "wasted_core_h"]].mean()
        CP.grid = lambda scenario, m=m: m
        open(f"{OUT}/sched_model_{sc}.tex", "w").write(CP.tab_grid(sc) + "\n")
        print("wrote sched_model_" + sc)


def main(parts=None):
    summary()
    model_grid()
    have = lambda f: os.path.exists(f"{P}/results/{f}")
    if have("grid_all.csv"):
        g = grid("grid_all.csv")
        policies(g, "sched_policies")
        loads(g, "sched_loads")
    if have("grid_llm.csv"):
        policies(grid("grid_llm.csv"), "sched_llm")
    if have("capacity_l1.2_512.csv"):
        capacity()
    if have("sens/sched_interval_30.csv"):
        sensitivity()
    if have("lisa/lisa_hybrid_32.csv"):
        lisa()


if __name__ == "__main__":
    main()
