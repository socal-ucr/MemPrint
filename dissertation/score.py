"""Analysis for dissertation/PROTOCOL.md.

    PYTHONPATH=analysis ~/environments/memfp/bin/python -W ignore dissertation/score.py [part ...]
Parts:
    plan   M3's runs (workload, split, config, interval) from M1's bytes fits -> $OUT/m3_list.txt
    m1     the TACO protocol on the taco, starts and bytes tables -> $OUT/m1/<set>/data/fits.csv, m1_summary.csv
    m8     bins and intervals per table, intervals dropped per workload -> $OUT/m8.csv
    m4     miniVite truths under three definitions and its six test MAPEs -> $OUT/m4.csv
    m3     sampler bins against splitter bins, the decision rule -> $OUT/m3_runs.csv, m3_mape.csv
    m6     Massif and DHAT peaks against the bytes truth and MemPrint -> $OUT/m6.csv
    m7     median wall time per configuration -> $OUT/m7.csv (and m7_k.csv for the matched-interval rerun)
    r1     the resident-memory model on MemPrint's inputs, scored as paper_rss's blind test -> $OUT/r1.csv, r1_splits.csv
"""
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "analysis"))
from memprint.dataset import prepare  # noqa: E402
from memprint.model import Model  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-11-05-diss-ch2")
PY = os.path.expanduser("~/environments/memfp/bin/python")
SETS = {"taco": os.path.expanduser("~/memory_estimator/tools/data"),
        "starts": os.path.join(REPO, "data/polybench-starts/data"),
        "bytes": os.path.join(REPO, "data/polybench-bytes/data")}
FINAL = f"{OUT}/m1/bytes-m4/data"   # bytes tables with M4's miniVite: the chapter's fits
WORKLOADS = open(os.path.join(REPO, "analysis/workloads.txt")).read().split()


def m1(sets=SETS):
    rows = []
    for name, src in sets.items():
        root = f"{OUT}/m1/{name}"
        os.makedirs(f"{root}/data", exist_ok=True)
        for w in WORKLOADS:
            subprocess.run(["cp", f"{src}/{w}_allData.csv", f"{root}/data/"], check=True)
        with open(f"{root}/build.log", "w") as log:
            subprocess.run([PY, "-W", "ignore", "-m", "memprint", "--root", root, "build", "--latex"], check=True,
                           stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": f"{REPO}/analysis"})
        f = pd.read_csv(f"{root}/data/fits.csv")
        g = f.groupby(["split", "subset"]).test_mape_train_si.agg(["mean", "median", "max", "count"]).reset_index()
        rows.append(g.assign(set=name))
    s = pd.concat(rows)
    s.to_csv(f"{OUT}/m1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["split", "subset"], columns="set", values=["mean", "max"]).round(2).to_string())


def plan():
    f = pd.read_csv(f"{OUT}/m1/bytes/data/fits.csv")
    lines = []
    for w in WORKLOADS:
        if w == "miniVite":   # after M4's retrace
            continue
        p = prepare(pd.read_csv(f"{OUT}/m1/bytes/data/{w}_allData.csv"))
        for split, config in (("EXTRA", p.last_config), ("INTER", p.middle_config)):
            g = f[(f.workload == w) & (f.split == split)]
            ks = set(g.train_si.astype(int)) | set(g.best_interval.dropna().astype(int))
            lines += [f"{w} {split} {config} {k}" for k in sorted(ks)]
    open(f"{OUT}/m3_list.txt", "w").write("\n".join(lines) + "\n")
    print(len(lines), "runs per sampler and seed")


def m8():
    rows = []
    for name, src in SETS.items():
        for w in WORKLOADS:
            d = pd.read_csv(f"{src}/{w}_allData.csv")
            bins = d[d.SamplingInterval > 1].groupby(["Config", "SamplingInterval"]).size()
            kept = prepare(d).intervals
            rows.append(dict(set=name, workload=w, bins_min=bins.min(), bins_max=bins.max(),
                             intervals=d.SamplingInterval[d.SamplingInterval > 1].nunique(), used=len(kept),
                             dropped=" ".join(str(k) for k in sorted(set(d.SamplingInterval) - set(kept) - {1}))))
    m = pd.DataFrame(rows)
    m.to_csv(f"{OUT}/m8.csv", index=False)
    print(m[m.set == "taco"].to_string(index=False))


def m4():
    taco = pd.read_csv(f"{SETS['taco']}/miniVite_allData.csv")
    old = pd.read_csv(f"{SETS['bytes']}/miniVite_allData.csv")
    new = pd.read_csv(f"{FINAL}/miniVite_allData.csv")
    t = lambda d: d[d.SamplingInterval == 1].groupby("Config").MemUsageObs.first()
    m = pd.DataFrame({"starts_B": t(taco), "bytes_alloc_B": t(old), "bytes_B": t(new)})
    m.index = m.index.astype(int)
    m = m.sort_index().assign(vs_alloc_pct=lambda x: (x.bytes_B / x.bytes_alloc_B - 1) * 100,
                              vs_starts_pct=lambda x: (x.bytes_B / x.starts_B - 1) * 100)
    m.to_csv(f"{OUT}/m4.csv")
    print(m.round(2).to_string())
    pub = {"EXTRA": [3.98, 4.98, 12.90], "INTER": [2.94, 1.04, 1.57]}
    f = pd.read_csv(f"{FINAL}/fits.csv")
    f = f[f.workload == "miniVite"].assign(published=lambda x: [pub[r.split][["NZ", "MT", "L2O"].index(r.subset)]
                                                                   for r in x.itertuples()])
    print(f[["split", "subset", "train_si", "test_mape_train_si", "best_interval", "test_mape_best",
             "published"]].round(2).to_string(index=False))


def sampler_bins(d):
    """MemUsageObs of each bin of one sampler run directory."""
    files = glob.glob(f"{d}/*_SubSample_*_bin_*.csv")
    return pd.Series([pd.read_csv(x).MemUsageObs.iloc[0] for x in files], dtype=float)


def m3(final=FINAL, done=None, tag=""):
    """done: (run directory, exit code) pairs, default $OUT/m3/done.txt; tag: suffix of the output CSVs."""
    fits = pd.read_csv(f"{final}/fits.csv")
    tables = {w: pd.read_csv(f"{final}/{w}_allData.csv") for w in WORKLOADS}
    runs, mapes = [], []
    for d, rc in done or (x.split() for x in open(f"{OUT}/m3/done.txt")):
        s, seed, w, config, k = d.split("/")[-5:]
        k = int(k)
        split = "EXTRA" if config == str(prepare(tables[w]).last_config) else "INTER"
        a = tables[w]
        a = a.assign(Config=a.Config.astype(str))
        truth = a[(a.SamplingInterval == 1) & (a.Config == config)].MemUsageObs.iloc[0]
        ref = a[(a.SamplingInterval == k) & (a.Config == config)]
        b = sampler_bins(d)
        r = dict(sampler=s, seed=int(seed), workload=w, split=split, config=config, k=k, rc=int(rc), bins=len(b),
                 zero_bins=int((b == 0).sum()), mean_ratio=b.mean() / ref.MemUsageObs.mean(),
                 sd_ratio=b.std() / ref.MemUsageObs.std())
        runs.append(r)
        if r["zero_bins"] or len(b) < 2:
            continue
        test = pd.DataFrame({"MemUsageObs": b, "SamplingInterval": k, "SD_MemUsage": b.std(), "Alpha": truth / b})
        for fit in fits[(fits.workload == w) & (fits.split == split)].itertuples():
            pred = Model.from_row(pd.Series(fit._asdict())).predict(test)
            sampler_mape = (pred.ErrorRate.abs()).mean()
            for basis, si, ref_mape in (("train_si", fit.train_si, fit.test_mape_train_si),
                                        ("best", fit.best_interval, fit.test_mape_best)):
                if si == k:
                    mapes.append(dict(sampler=s, seed=int(seed), workload=w, split=split, subset=fit.subset,
                                      basis=basis, k=k, sampler_mape=sampler_mape, splitter_mape=ref_mape))
    runs, mapes = pd.DataFrame(runs), pd.DataFrame(mapes)
    runs.to_csv(f"{OUT}/m3_runs{tag}.csv", index=False)
    mapes.to_csv(f"{OUT}/m3_mape{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("runs", len(runs), "failed", int((runs.rc != 0).sum()), "with a zero bin", int((runs.zero_bins > 0).sum()))
    print(runs.groupby("sampler")[["mean_ratio", "sd_ratio"]].describe(percentiles=[.1, .5, .9]).T.round(3))
    g = mapes.groupby(["sampler", "basis", "split", "subset", "workload"])[["sampler_mape", "splitter_mape"]].mean()
    g["diff"] = g.sampler_mape - g.splitter_mape
    print(g.groupby(["sampler", "basis", "split", "subset"]).agg(
        sampler_mean=("sampler_mape", "mean"), splitter_mean=("splitter_mape", "mean"),
        diff_median=("diff", "median"), n=("diff", "size")).round(2).to_string())
    b = g.loc[("boot", "train_si")]
    sd = runs[runs.sampler == "boot"].sd_ratio.median()
    disj = b["diff"].median() > 2 or not 0.9 <= sd <= 1.1
    print(f"decision: boot diff median {b['diff'].median():.2f} points, SD ratio median {sd:.3f}"
          f" -> report {'disjoint' if disj else 'bootstrap'}")


M11 = f"{OUT}/m11"


def m11build(root=M11):
    """M11: the pages tables from the splitter traces (m11.sh trace), M1's analysis on them -> $OUT/m11/data/fits.csv,
    and M3's run list from the pages fits -> $OUT/m11/m3_list.txt. B50 uses the same steps on its own root."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}/analysis"}
    for cmd in ("preprocess", "build"):
        with open(f"{root}/{cmd}.log", "w") as log:
            subprocess.run([PY, "-W", "ignore", "-m", "memprint", "--root", root, cmd] + (["--latex"] if cmd == "build" else []),
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    f = pd.read_csv(f"{root}/data/fits.csv")
    lines = []
    for w in WORKLOADS:
        p = prepare(pd.read_csv(f"{root}/data/{w}_allData.csv"))
        for split, config in (("EXTRA", p.last_config), ("INTER", p.middle_config)):
            g = f[(f.workload == w) & (f.split == split)]
            ks = set(g.train_si.astype(int)) | set(g.best_interval.dropna().astype(int))
            lines += [f"{w} {split} {config} {k}" for k in sorted(ks)]
    open(f"{root}/m3_list.txt", "w").write("\n".join(lines) + "\n")
    print(len(lines), "runs per sampler and seed")


def m11():
    """M11: the TACO protocol on pages touched next to bytes touched (M1), and M3's sampler scoring on the pages
    tables -> $OUT/m11_summary.csv, m3_runs_pages.csv, m3_mape_pages.csv."""
    rows = []
    for name, root in (("bytes", FINAL), ("pages", f"{M11}/data")):
        f = pd.read_csv(f"{root}/fits.csv")
        g = f.groupby(["split", "subset"]).test_mape_train_si.agg(["mean", "median", "max", "count"]).reset_index()
        rows.append(g.assign(set=name))
        f.assign(set=name).to_csv(f"{OUT}/m11_fits_{name}.csv", index=False)
    s = pd.concat(rows)
    s.to_csv(f"{OUT}/m11_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["split", "subset"], columns="set", values=["mean", "median", "max"]).round(2).to_string())
    a = pd.read_csv(f"{OUT}/m11_fits_bytes.csv").set_index(["workload", "split", "subset"]).test_mape_train_si
    b = pd.read_csv(f"{OUT}/m11_fits_pages.csv").set_index(["workload", "split", "subset"]).test_mape_train_si
    d = pd.DataFrame({"bytes": a, "pages": b}).assign(diff=lambda x: x.pages - x.bytes)
    print(d[d.index.get_level_values("subset") == "L2O"].round(2).to_string())
    rc = {}
    for line in open(f"{M11}/done.txt"):
        x = line.split()
        if x[0] in ("boot", "disj"):
            rc[f"{M11}/m3/{x[0]}/{x[3]}/{x[1]}/{x[2]}/{x[4]}"] = x[5]
    m3(f"{M11}/data", [(p, rc.get(p, "1")) for p in sorted(glob.glob(f"{M11}/m3/*/*/*/*/*"))], "_pages")


B50 = os.path.expanduser("~/memprint-data/2026-11-11-bins50")


def b50build():
    """B50: the 50-bin tables from dissertation/bins50.sh trace, the refit, and M3's run list from the new fits."""
    m11build(B50)


def b50():
    """B50: test MAPE with 50 bins against 20 (M1's bytes tables), the intervals kept, and M3's samplers on the
    50-bin tables (bins50.sh sample) -> $OUT/b50_summary.csv, b50_fits.csv, m3_runs_b50.csv, m3_mape_b50.csv."""
    rows, fits = [], []
    for name, root in (("20", FINAL), ("50", f"{B50}/data")):
        f = pd.read_csv(f"{root}/fits.csv").assign(bins=name)
        fits.append(f)
        g = f.groupby(["split", "subset"]).test_mape_train_si.agg(["mean", "median", "max", "count"]).reset_index()
        rows.append(g.assign(bins=name))
    s = pd.concat(rows)
    s.to_csv(f"{OUT}/b50_summary.csv", index=False)
    f = pd.concat(fits)
    f.to_csv(f"{OUT}/b50_fits.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["split", "subset"], columns="bins", values=["mean", "median", "max"]).round(2).to_string())
    w = f[f.subset == "L2O"].pivot_table(index=["workload", "split"], columns="bins", values="test_mape_train_si")
    print(w.assign(diff=w["50"] - w["20"]).round(2).sort_values("diff").to_string())
    kept = []
    for b, root in (("20", FINAL), ("50", f"{B50}/data")):
        for wl in WORKLOADS:
            kept.append(dict(bins=b, workload=wl, intervals=len(prepare(pd.read_csv(f"{root}/{wl}_allData.csv")).intervals)))
    k = pd.DataFrame(kept).pivot(index="workload", columns="bins", values="intervals")
    k.to_csv(f"{OUT}/b50_intervals.csv")
    print("intervals kept: 20 bins", k["20"].sum(), "50 bins", k["50"].sum(), "; fewer than 13:",
          (k["20"] < 13).sum(), "and", (k["50"] < 13).sum())
    if os.path.exists(f"{B50}/done.txt"):
        rc = {}
        for line in open(f"{B50}/done.txt"):
            x = line.split()
            if x[0] in ("boot", "disj"):
                rc[f"{B50}/m3/{x[0]}/{x[3]}/{x[1]}/{x[2]}/{x[4]}"] = x[5] if int(x[6]) > 0 else "1"
        runs = sorted(glob.glob(f"{B50}/m3/*/*/*/*/*"))
        if runs:
            m3(f"{B50}/data", [(p, rc.get(p, "1")) for p in runs], "_b50")


def m6():
    rows = []
    for x in sorted(glob.glob(f"{OUT}/m6/*.massif")):
        name = os.path.basename(x)[:-7]
        w, c = name.rsplit("-", 1)
        txt = open(x).read()
        heap = [int(h) + int(e) for h, e in re.findall(r"mem_heap_B=(\d+)\nmem_heap_extra_B=(\d+)", txt)]
        useful = [int(h) for h in re.findall(r"mem_heap_B=(\d+)", txt)]
        dhat = json.load(open(f"{OUT}/m6/{name}.dhat"))
        a = pd.read_csv(f"{FINAL}/{w}_allData.csv")
        truth = a[(a.SamplingInterval == 1) & (a.Config.astype(str) == c)].MemUsageObs.iloc[0]
        rows.append(dict(workload=w, config=c, truth_B=truth, massif_B=max(useful), massif_extra_B=max(heap),
                         dhat_B=sum(p["gb"] for p in dhat["pps"])))
    m = pd.DataFrame(rows)
    for c in ("massif_B", "dhat_B"):
        m[c.replace("_B", "_over_truth")] = m[c] / m.truth_B
    m.to_csv(f"{OUT}/m6.csv", index=False)
    print(m.round(3).to_string(index=False))


def m7():
    for src, dst in (("timing.csv", "m7.csv"), ("timing_k.csv", "m7_k.csv")):
        if not os.path.exists(f"{OUT}/m7/{src}"):
            continue
        t = pd.read_csv(f"{OUT}/m7/{src}")
        m = t.groupby("config").seconds.agg(["median", "min", "max", "count"])
        m["slowdown"] = m["median"] / m.loc["native", "median"]
        m.to_csv(f"{OUT}/{dst}")
        print(src)
        print(m.sort_values("median").round(3).to_string())


def input_size(w, config):
    """Elements of the arrays a PolyBench kernel declares at a dataset (its header's sizes); vertices for miniVite."""
    if w == "miniVite":
        return int(config)
    src = os.path.join(REPO, "workloads/src/polybench")
    c = subprocess.run(["find", src, "-name", f"{w}.c"], capture_output=True, text=True).stdout.split()[0]
    h = open(c[:-2] + ".h").read()
    block = re.search(rf"#\s*ifdef {config}_DATASET\n(.*?)#\s*endif", h, re.S).group(1)
    dims = {k: v for k, v in re.findall(r"#\s*define (\w+) (.+)", block)}
    for _ in range(3):
        dims = {k: str(eval(v, {}, {a: int(b) for a, b in dims.items() if b.isdigit()})) for k, v in dims.items()}
    total = 0
    for args in re.findall(r"POLYBENCH_\dD_ARRAY_DECL\((.*?)\);", open(c).read()):
        n = 1
        sizes = args.split(",")[2:]
        for d in sizes[:len(sizes) // 2]:
            n *= int(eval(d, {}, {k: int(v) for k, v in dims.items()}))
        total += n
    return total


def inuse_truth(c):
    """R11: miniVite's largest number of pages in use (frees tracked) over the run, in bytes."""
    t = pd.read_csv(glob.glob(f"{OUT}/r11/{c}/Buffered_*_timeline.csv")[0])
    return float(t[t.Bin == -1].MemUsageObs.max())


def r1(sampler="pss1", tag="", truth_src="bytes", inuse=False):
    """(a) the largest anon - free of the run, median of 3; (b) (a) + R, R the median of truth - (a) over the other
    workloads' configs (paper_rss/blind/score.py)."""
    import numpy as np
    rows = []
    for w in WORKLOADS:
        a = pd.read_csv(f"{FINAL}/{w}_allData.csv")
        p = prepare(a)
        for c in p.configs:
            peaks = []
            for rep in (1, 2, 3):
                r = pd.read_csv(f"{OUT}/r1/{w}/{c}/{rep}/{sampler}.csv")
                peaks.append((r.anon_bytes - r.free_bytes).clip(lower=0).max())
            t = p.truth[c] if truth_src == "bytes" else pages_truth(w, c)
            if inuse and w == "miniVite":
                t = inuse_truth(c)
            rows.append(dict(workload=w, config=str(c), truth=t, bytes_truth=p.truth[c], measured=float(np.median(peaks)),
                             split="EXTRA" if c == p.last_config else "INTER" if c == p.middle_config else ""))
    m = pd.DataFrame(rows)
    m["R"] = [float((m[m.workload != w].truth - m[m.workload != w].measured).median()) for w in m.workload]
    m["err_a"] = 100 * (m.measured - m.truth) / m.truth
    m["err_b"] = 100 * (m.measured + m.R - m.truth) / m.truth
    m.to_csv(f"{OUT}/r1{tag}.csv", index=False)
    s = m[m.split != ""].copy()
    fits = pd.read_csv(f"{FINAL}/fits.csv")
    l2o = fits[fits.subset == "L2O"].set_index(["workload", "split"]).test_mape_train_si
    sam = pd.read_csv(f"{OUT}/m3_mape.csv")
    sam = sam[(sam.sampler == "boot") & (sam.basis == "train_si") & (sam.subset == "L2O")]
    sam = sam.groupby(["workload", "split"]).sampler_mape.mean()
    s["memprint_splitter"] = [l2o.get((r.workload, r.split)) for r in s.itertuples()]
    s["memprint_sampler"] = [sam.get((r.workload, r.split)) for r in s.itertuples()]
    # secondary: the transfer rule of paper_rss (blind/score.py) from the other configs, size = array elements
    sys.path.insert(0, os.path.join(REPO, "paper_rss", "blind"))
    from score import transfer
    m["program"], m["size"] = m.workload, [input_size(r.workload, r.config) for r in m.itertuples()]
    for split in ("EXTRA", "INTER"):
        for r in s[s.split == split].itertuples():
            pred = transfer(m, r.workload, r.config, floor_smaller=(split == "INTER")) + r.R
            s.loc[r.Index, "transfer_err"] = 100 * (pred - r.truth) / r.truth
    s.to_csv(f"{OUT}/r1{tag}_splits.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))
    s["abs_a"], s["abs_b"] = s.err_a.abs(), s.err_b.abs()
    s["abs_transfer"] = s.transfer_err.abs()
    print(s.groupby("split")[["abs_a", "abs_b", "abs_transfer", "memprint_splitter", "memprint_sampler"]].agg(["mean", "median"]).round(2))
    print("all configs, median |err| a, b:", round(m.err_a.abs().median(), 2), round(m.err_b.abs().median(), 2))


def m9():
    """Median wall time over 5 rounds per workload and run, and the sampler's time over the truth run's."""
    t = pd.read_csv(f"{OUT}/m9/timing.csv")
    m = t.groupby(["workload", "config"]).seconds.median().unstack()
    m = m[["native", "instr", "truth", "splitter", "boot-1000", "disj-1000"]]
    for c in ("boot-1000", "disj-1000"):
        m[f"truth_over_{c}"] = m.truth / m[c]
    m["truth_over_instr"] = m.truth / m.instr
    m.to_csv(f"{OUT}/m9.csv")
    print(t.groupby(["workload", "config"]).seconds.agg(["count", "min", "max"]).round(2).unstack().to_string())
    print(m.round(2).to_string())


def pages_truth(w, c):
    """R9's pages-touched truth of an R1 input (dissertation/r9pages.sh)."""
    import glob
    f = [x for x in glob.glob(f"{OUT}/r9/all/pages/{w}/{c}/Buffered_*.csv") if "SubSample" not in x]
    return float(pd.read_csv(f[0]).MemUsageObs.iloc[0])


def r1p():
    """R1 against the pages-touched truth (primary for Ch. 3), under the 1 ms and the event sampler."""
    r1("pss1", "_pages", "pages")
    r1("event", "_r8_pages", "pages")


def first_sample(w, c, sampler):
    """The in-run base of an R1 input, median of 3 runs: anon - free (base0) and RSS - anon (file0, Ch. 3's) at the
    first sample."""
    import numpy as np
    b0, f0 = [], []
    for rep in (1, 2, 3):
        r = pd.read_csv(f"{OUT}/r1/{w}/{c}/{rep}/{sampler}.csv")
        r = r[r.rss_bytes > 0]
        b0.append(max(r.anon_bytes.iloc[0] - r.free_bytes.iloc[0], 0)); f0.append(r.rss_bytes.iloc[0] - r.anon_bytes.iloc[0])
    return float(np.median(b0)), float(np.median(f0))


def r10():
    """R10: the in-run base in place of R, and subtracted before the transfer rule scales. R1 (pages and bytes truths)
    under both samplers from r1{tag}.csv, and the blind configurations (paper_rss/blind) under both samplers."""
    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "paper_rss", "blind"))
    import score as B
    rows, out = [], []
    for sampler, tag in (("event", "_r8_pages"), ("pss1", "_pages")):
        m = pd.read_csv(f"{OUT}/r1{tag}.csv", dtype={"config": str})
        m[["base0", "file0"]] = [first_sample(r.workload, r.config, sampler) for r in m.itertuples()]
        m["program"], m["size"] = m.workload, [input_size(r.workload, r.config) for r in m.itertuples()]
        for truth in ("pages", "bytes"):
            t = m.truth if truth == "pages" else m.bytes_truth
            mm = m.assign(truth=t)
            preds = {}
            for name, Bc in (("R", None), ("file0", "file0"), ("base0", "base0")):
                b = 0 if Bc is None else mm[Bc]
                c = [float((mm.truth - mm.measured - b)[mm.workload != w].median()) for w in mm.workload]
                preds[name] = mm.measured + b + c
                if Bc == "file0":
                    preds["file0 alone"] = mm.measured + b
            for name, p in preds.items():
                mm[f"err_{name}"] = 100 * (p - mm.truth) / mm.truth
            for r in mm[mm.split.notna()].itertuples():
                d = dict(sampler=sampler, truth=truth, workload=r.workload, config=r.config, split=r.split,
                         base0=r.base0, file0=r.file0)
                for name in preds:
                    d[f"b_{name}"] = mm.loc[r.Index, f"err_{name}"]
                for base in (None, "base0"):
                    pred = B.transfer(mm, r.workload, r.config, floor_smaller=(r.split == "INTER"), base=base) + r.R \
                        if truth == "pages" else None
                    if truth == "bytes":
                        Rb = float((mm.truth - mm.measured)[mm.workload != r.workload].median())
                        pred = B.transfer(mm, r.workload, r.config, floor_smaller=(r.split == "INTER"), base=base) + Rb
                    d["c_" + (base or "plain")] = 100 * (pred - r.truth if truth == "pages" else pred - r.bytes_truth) / \
                        (r.truth if truth == "pages" else r.bytes_truth)
                rows.append(d)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/r10_r1.csv", index=False)
    cols = [c for c in d.columns if c.startswith(("b_", "c_"))]
    pd.set_option("display.width", 220)
    print(d.groupby(["sampler", "truth", "split"])[cols].agg(lambda x: x.abs().median()).round(2).to_string())
    print(d.groupby(["sampler", "truth", "split"])[cols].agg(lambda x: x.abs().mean()).round(2).to_string())
    print(d[(d.workload == "miniVite")].round(1).to_string(index=False))
    # blind configurations
    RS = os.path.expanduser("~/memprint-data/2026-11-05-rsssample/blind")
    bl = []
    for sampler, runs, name in (("pss1", B.RUNS, "{rep}.csv"), ("event", f"{RS}/event", "{rep}/event.csv")):
        m = B.measured(runs, name).merge(B.truths(), on=["program", "config"], how="left")
        m["R"] = [float((m[(m.program != p) & m.truth.notna()].truth - m[(m.program != p) & m.truth.notna()].measured).median())
                  for p in m.program]
        for prog in B.SIZE:
            have = m[(m.program == prog) & m.truth.notna()].sort_values("size")
            for split, target in (("EXTRA", have.config.iloc[-1]), ("INTER", B.INTER.get(prog))):
                if target is None:
                    continue
                row = m[(m.program == prog) & (m.config == target)].iloc[0]
                e = {}
                for base in (None, "base0"):
                    pred = B.transfer(m, prog, target, base=base) + row.R
                    e["c_" + (base or "plain")] = 100 * (pred - row.truth) / row.truth
                bl.append(dict(sampler=sampler, program=prog, split=split, config=target, **e,
                               base0_src=have[have.config != target].base0.iloc[-1], base0=row.base0,
                               measured_src=have[have.config != target].measured.iloc[-1]))
    bl = pd.DataFrame(bl)
    bl.to_csv(f"{OUT}/r10_blind.csv", index=False)
    print(bl.round(1).to_string(index=False))


def r11():
    """R11: R1 with miniVite's truth the pages in use at the peak (the other workloads keep the pages truth)."""
    rows = [dict(config=c, touched=pages_truth("miniVite", c), inuse=inuse_truth(c)) for c in (1024, 2048, 4096, 8192, 16384)]
    d = pd.DataFrame(rows).assign(ratio=lambda x: x.touched / x.inuse)
    d.to_csv(f"{OUT}/r11.csv", index=False)
    print(d.to_string(index=False))
    r1("event", "_r11", "pages", inuse=True)
    r1("pss1", "_r11_pss1", "pages", inuse=True)


def r8():
    """R1 under the event sampler (run.sh part r8)."""
    r1("event", "_r8")


if __name__ == "__main__":
    for part in sys.argv[1:] or ["m1", "plan", "m8"]:
        print("==", part)
        globals()[part]()
