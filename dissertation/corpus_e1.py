"""Full E1: MemPrint's signed error on each scheduling-corpus program at its largest input (corpus_e1.sh).

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/corpus_e1.py build   (after split)
    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/corpus_e1.py errors  (after truth, sample)

build   per program: the splitter traces of its two smaller inputs -> $E/data/<prog>_allData.csv; the L2O model
        (NZ when the two inputs give no monotone interval) fitted on both, its training-best interval ->
        $E/fits.csv and $E/sample_list.txt ("<prog> <largest input> <k>")
errors  the bootstrap sampler's bins at the largest input under that model: e = mean over bins of predicted alpha x
        bin / truth - 1, the truth the one-bin run's bytes touched -> $E/mp_errors_corpus.csv (the columns of
        dissertation/mp_errors.py, workload = the corpus program), for real.make_jobs (opts mp_own)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, HERE)
from memprint import traces  # noqa: E402
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model, mape_by_interval  # noqa: E402

E = os.path.expanduser("~/memprint-data/2026-11-23-corpus-e1")
LISTS = [os.path.expanduser(f"~/memprint-data/2026-10-13-sched-llm/corpus/{f}") for f in ("list.txt", "list_llm.txt")]


def inputs(prog):
    return [l.split()[1] for f in LISTS for l in open(f) if l.split() and l.split()[0] == prog]


def build():
    os.makedirs(f"{E}/data", exist_ok=True)
    fits, plan = [], []
    for prog in sorted(os.listdir(f"{E}/traces")):
        t = traces.add_sample_spread(traces.load_traces(f"{E}/traces/{prog}"))
        if t.Config.nunique() < 2:
            print(prog, "has", t.Config.nunique(), "training input(s): skipped")
            continue
        t.to_csv(f"{E}/data/{prog}_allData.csv", index=False)
        p = prepare(t)
        sub = "L2O" if p.best_interval is not None else "NZ"
        d = p.subset(sub)
        m = Model.fit(d)
        tm = mape_by_interval(m.predict(d))
        k = int(tm.idxmin())
        big = inputs(prog)[2]
        fits.append(dict(workload=prog, subset=sub, train_si=k, train_mape=tm.min(), best_interval=p.best_interval,
                         configs=" ".join(map(str, p.configs)), test_config=big,
                         **dict(zip(["intercept"] + [f"b{i}" for i in range(1, 8)], m.coefficients))))
        plan.append(f"{prog} {big} {k}")
    pd.DataFrame(fits).to_csv(f"{E}/fits.csv", index=False)
    open(f"{E}/sample_list.txt", "w").write("\n".join(plan) + "\n")
    print(pd.DataFrame(fits)[["workload", "subset", "train_si", "train_mape", "best_interval", "configs", "test_config"]]
          .round(2).to_string(index=False))


def truth(prog, inp):
    f = [x for x in glob.glob(f"{E}/truth/{prog}/Buffered_{prog}-{inp}_*.csv") if "SubSample" not in x and "timeline" not in x]
    t = pd.read_csv(f[0])
    return float(t[(t.FunctionName == "Total") & (t.SamplingInterval == 1)].MemUsageObs.iloc[0])


def errors(floor=False):
    """floor: alpha_i clipped at 1 (a bin is a subset of the footprint, so the true alpha is at least 1)
    -> $E/mp_errors_corpus_floor.csv"""
    from metric_check import errors as err
    if floor:
        orig = Model.predict
        def clipped(self, d):
            p = orig(self, d)
            return p.assign(Predicted_Alpha=p.Predicted_Alpha.clip(lower=1))
        Model.predict = clipped
    fits = pd.read_csv(f"{E}/fits.csv")
    rows = []
    for f in fits.itertuples():
        T = truth(f.workload, f.test_config)
        model = Model.from_row(pd.Series(f._asdict()))
        for seed in (1, 2, 3):
            files = glob.glob(f"{E}/m3/boot/{seed}/{f.workload}/{f.test_config}/{f.train_si}/*_SubSample_*_bin_*.csv")
            if not files:
                print("missing", f.workload, seed)
                continue
            b = [pd.read_csv(x).MemUsageObs.iloc[0] for x in files]
            e = err(model, b, f.train_si, T)
            rows.append(dict(sampler="boot", workload=f.workload, seed=seed, k=f.train_si, truth=T, e=e["mean_est"] / 100,
                             per_bin_mape=e["per_bin"], subset=f.subset))
    d = pd.DataFrame(rows)
    d.to_csv(f"{E}/mp_errors_corpus{'_floor' if floor else ''}.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["workload", "subset"]).e.agg(["min", "mean", "max"]).mul(100).round(1).to_string())
    print(f"{d.workload.nunique()} programs, {len(d)} runs; |e| median {d.e.abs().median() * 100:.2f}%, mean "
          f"{d.e.abs().mean() * 100:.2f}%; under-estimates {(d.e < 0).mean() * 100:.0f}%; within a factor of 2: "
          f"{((1 + d.e > 0.5) & (1 + d.e < 2)).mean() * 100:.0f}% of runs")


def errors_floor():
    errors(floor=True)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
