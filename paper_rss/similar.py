"""Does code similarity predict the similarity of the RSS model's terms? (paper_rss, the generalisation section)

    PYTHONPATH=analysis ~/environments/memfp/bin/python paper_rss/similar.py
        ->  paper_rss/data/similar_{features,terms,mantel,lopo,beta}.csv

Programs: the corpus programs with code of their own (similar_src.PROGRAMS: all but cc1).
Terms, per program at its largest input, from its own runs (the quantities that zero.py and borrow.py take from
other programs):
  B     the runtime's pages (compute.base_of), MB
  r     the allocator's free share of anonymous RSS at the model run's peak (median of 3 runs)
  R'    RSS peak less the largest data estimate A - F, the libraries' and runtime's pages (median of 3 runs), MB
  beta  the exponent of the data peak in the input size (borrow.beta, the --middle setting; programs with two
        inputs or more)
Code spaces:
  static   features read from the code and the executable, without running it (features() below), standardised
           over the known programs, Euclidean distance
  qodo, bgecode, gte, qwen3   code embeddings (similar_embed.py), cosine distance; <model>_mem: of the memory code
           alone (similar_src.mem_slice)
  descriptor   borrow.py's descriptor of the program's run at its smaller input (a run, not code: the reference)
Two tests:
  Mantel   Spearman correlation over program pairs of the code distance and the terms' distance |a - b| (B and R'
           on a log scale), with a permutation p-value (10,000 permutations of the programs)
  LOPO     each program's term predicted from the other programs: their median (the baseline, as zero.py), the
           nearest program in the space, the mean of the 3 nearest, and the RBF weighting of borrow.py; the error is
           |predicted - own|. For beta, also the whole-run error of borrow.predict with the predicted beta.
Static features, counted on the code with comments and strings removed, per 1,000 lines where a rate:
  log lines; allocation calls (malloc, calloc, aligned/posix_memalign, new, PolyBench's array allocation; Lua's
  table constructors, Perl's hash and array stores, Python's dict and list literals, SQL INSERT); realloc; releases
  (free, delete, PolyBench's free; Lua's = nil, Perl's delete and undef, Python's del, SQL DELETE and DROP); the
  share of allocations inside a loop; containers that grow (push_back, emplace_back, resize, reserve, insert,
  append, push, table.insert); node-based containers (std::map, set, list, deque, unordered_*; hashes and dicts);
  std::vector; smart pointers; OpenMP pragmas; MPI calls; and from the executable: log text, data and bss sizes,
  the number of shared libraries and log of their total size, and whether it links libstdc++, libgomp, MPI or is a
  language interpreter.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv.append("--middle")   # borrow.py's post-hoc setting: the PolyBench smallest inputs are empty
import borrow as BR  # noqa: E402
import compute as M  # noqa: E402
import similar_src as S  # noqa: E402

EMB = os.path.expanduser("~/memprint-data/2026-10-29-rsssim")
SPACES = ("static", "static_src", "static_bin", "qodo", "bgecode", "gte", "qwen3", "qodo_mem", "bgecode_mem",
          "gte_mem", "qwen3_mem", "descriptor")
BIN = ["log_text", "log_data", "log_bss", "n_libs", "log_libs", "stdcxx", "gomp", "mpi", "interp"]
TERMS = ("B", "r", "Rp", "beta")
LOG = {"B", "Rp"}
RNG = np.random.default_rng(0)

PATTERNS, COMMON, strip = S.PATTERNS, S.COMMON, S.strip


def in_loop_share(code, alloc):
    """The share of allocation calls inside a for, while or do loop (C and C++), from the braces."""
    tok = re.compile(r"\b(for|while|do)\b|[(){};]|" + alloc)
    stack, pending, paren, n, inside = [], False, 0, 0, 0
    for m in tok.finditer(code):
        t = m.group(0)
        if m.group(1):
            pending, paren = True, 0
        elif t == "(":
            paren += 1
        elif t == ")":
            paren = max(paren - 1, 0)
        elif t == "{":
            stack.append(pending and paren == 0); pending = False
        elif t == "}":
            if stack: stack.pop()
        elif t == ";":
            if paren == 0:
                pending = False   # a loop over a single statement ends here
        else:   # an allocation
            n += 1
            inside += any(stack) or (pending and paren == 0)
    return inside / n if n else 0.0


def binary(exe):
    out = subprocess.run(["size", exe], capture_output=True, text=True).stdout.splitlines()[1].split()
    text, data, bss = (int(v) for v in out[:3])
    libs = re.findall(r"=>\s*(\S+)", subprocess.run(["ldd", exe], capture_output=True, text=True).stdout)
    libs = [l for l in libs if os.path.exists(l)]
    names = " ".join(os.path.basename(l) for l in libs)
    return dict(log_text=np.log10(text + 1), log_data=np.log10(data + 1), log_bss=np.log10(bss + 1),
                n_libs=len(libs), log_libs=np.log10(sum(os.path.getsize(os.path.realpath(l)) for l in libs) + 1),
                stdcxx=float("libstdc++" in names), gomp=float("libgomp" in names),
                mpi=float("libmpi" in names), interp=float(os.path.basename(exe) in ("lua", "perl", "python3", "sqlite3")))


def features():
    rows = []
    for p, (lang, files, exe) in S.PROGRAMS.items():
        code = strip(S.text(p), lang)
        kloc = max(sum(1 for l in code.splitlines() if l.strip()), 1) / 1000
        pat = {**COMMON, **PATTERNS[lang]}
        cnt = {k: len(re.findall(v, code)) for k, v in pat.items()}
        f = dict(prog=p, log_lines=np.log10(kloc * 1000), **{f"{k}_per_kloc": v / kloc for k, v in cnt.items()},
                 free_over_alloc=cnt["free"] / max(cnt["alloc"], 1),
                 alloc_in_loop=in_loop_share(code, PATTERNS[lang]["alloc"]) if lang in ("c", "cpp") else 0.0,
                 **binary(exe))
        rows.append(f)
    return pd.DataFrame(rows).set_index("prog")


def terms():
    ins = M.inputs()
    P = BR.programs()
    rows = []
    for p in S.PROGRAMS:
        inp = ins[p]
        r, rest = [], []
        for rep in (1, 2, 3):
            o = f"{M.O}/runs_all/{p}/{rep}"
            b = pd.read_csv(f"{o}/base2.csv"); b = b[b.rss_bytes > 0]
            s = pd.read_csv(f"{o}/pss.csv"); s = s[s.rss_bytes > 0]
            k = int(np.argmax(s.anon_bytes))
            r.append(s.free_bytes.iloc[k] / max(s.anon_bytes.iloc[k], 1))
            rest.append(b.rss_bytes.max() - (s.anon_bytes - s.free_bytes).max())
        rows.append(dict(prog=p, B=M.base_of(p, inp) / 1e6, r=np.median(r), Rp=np.median(rest) / 1e6,
                         beta=BR.beta(P[p]) if p in P else np.nan))
    return pd.DataFrame(rows).set_index("prog"), P


def columns(F, space):
    """The static features of a space: all of them, the code's alone (static_src) or the executable's (static_bin)."""
    return F[BIN] if space == "static_bin" else F.drop(columns=BIN) if space == "static_src" else F


def distances(space, F, progs, P):
    """The full distance matrix of the programs in a space (static: standardised over all programs; LOPO
    restandardises below)."""
    if space.startswith("static"):
        G = columns(F, space).loc[progs]
        Z = ((G - G.mean()) / G.std().replace(0, 1)).to_numpy(float)
        return np.linalg.norm(Z[:, None] - Z[None], axis=2)
    if space == "descriptor":
        Z = np.array([BR.descriptor(P[p]) if p in P else np.full(6, np.nan) for p in progs])
        Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-12)
        return np.linalg.norm(Z[:, None] - Z[None], axis=2)
    d = np.load(f"{EMB}/emb_{space}.npz")
    E = dict(zip(d["progs"], d["emb"]))
    X = np.array([E[p] for p in progs])
    return 1 - X @ X.T


def static_d(F, known, c):
    """Distances from c to the known programs, standardised over the known programs (c is not seen)."""
    z = F.loc[known].to_numpy(float)
    mu, sd = z.mean(0), z.std(0)
    keep = sd > 0
    zs = (z[:, keep] - mu[keep]) / sd[keep]
    zc = (F.loc[c].to_numpy(float)[keep] - mu[keep]) / sd[keep]
    return np.linalg.norm(zs - zc, axis=1), zs


def mantel(D, y, log):
    v = np.log10(y) if log else y
    iu = np.triu_indices(len(v), 1)
    dy = np.abs(v[:, None] - v[None])
    rho = spearmanr(D[iu], dy[iu])[0]
    perm = []
    for _ in range(10000):
        q = RNG.permutation(len(v))
        perm.append(spearmanr(D[iu], dy[q][:, q][iu])[0])
    return rho, (1 + np.sum(np.array(perm) >= rho)) / 10001


def rbf_weights(d, dk):
    """borrow.py's RBF: h the median distance of a known program to its nearest other known program."""
    nn = np.sort(dk + np.diag(np.full(len(dk), np.inf)), axis=1)[:, 0]
    h = np.median(nn)
    w = np.exp(-(d ** 2 - d.min() ** 2) / (2 * h ** 2))
    return w / w.sum()


def main():
    F = features()
    T, P = terms()
    os.makedirs(f"{HERE}/data", exist_ok=True)
    F.to_csv(f"{HERE}/data/similar_features.csv"); T.to_csv(f"{HERE}/data/similar_terms.csv")
    progs = list(S.PROGRAMS)
    full = {s: distances(s, F, progs, P) for s in SPACES
            if s.startswith("static") or s == "descriptor" or os.path.exists(f"{EMB}/emb_{s}.npz")}
    mrows, lrows, brows = [], [], []
    for term in TERMS:
        have = [p for p in progs if not np.isnan(T.loc[p, term])]
        idx = [progs.index(p) for p in have]
        y = T.loc[have, term].to_numpy(float)
        for s, D in full.items():
            sub = D[np.ix_(idx, idx)]
            if np.isnan(sub).any():
                continue
            rho, pv = mantel(sub, y, term in LOG)
            mrows.append(dict(term=term, space=s, n=len(have), rho=rho, p=pv))
        for i, c in enumerate(have):
            known = [p for p in have if p != c]
            yk = np.delete(y, i)
            preds = {"median": np.median(yk)}
            for s, D in full.items():
                if s.startswith("static"):
                    d, zs = static_d(columns(F, s), known, c)
                    dk = np.linalg.norm(zs[:, None] - zs[None], axis=2)
                else:
                    ki = [progs.index(p) for p in known]
                    d, dk = D[progs.index(c), ki], D[np.ix_(ki, ki)]
                    if np.isnan(d).any():
                        continue
                o = np.argsort(d)
                preds[f"{s}:nearest"] = yk[o[0]]
                preds[f"{s}:knn3"] = yk[o[:3]].mean()
                preds[f"{s}:rbf"] = float(rbf_weights(d, dk) @ yk)
                if term == "beta":
                    brows.append(dict(prog=c, space=s, nearest=known[o[0]]))
            for m, v in preds.items():
                lrows.append(dict(term=term, prog=c, method=m, pred=v, own=y[i], err=abs(v - y[i])))
                if term == "beta":
                    p = P[c]
                    t, base = M.truth(c, p["target"]), M.base_of(c, p["target"])
                    e = M.err(BR.predict(p, v) + base, t)
                    lrows[-1].update(whole_signed=e["signed"], whole_abs=e["abs"])
            if term == "beta":
                p = P[c]
                t, base = M.truth(c, p["target"]), M.base_of(c, p["target"])
                for m, b in (("proportional", 1.0), ("own_beta", y[i])):
                    e = M.err(BR.predict(p, b) + base, t)
                    lrows.append(dict(term=term, prog=c, method=m, pred=b, own=y[i], err=abs(b - y[i]),
                                      whole_signed=e["signed"], whole_abs=e["abs"]))
    mt, lo, nb = pd.DataFrame(mrows), pd.DataFrame(lrows), pd.DataFrame(brows)
    mt.to_csv(f"{HERE}/data/similar_mantel.csv", index=False)
    lo.to_csv(f"{HERE}/data/similar_lopo.csv", index=False)
    nb.to_csv(f"{HERE}/data/similar_beta_nearest.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string())
    print(mt.pivot(index="space", columns="term", values=["rho", "p"]).round(3).to_string())
    print(lo.groupby(["term", "method"]).err.median().unstack(0).round(3).to_string())
    b = lo[lo.term == "beta"]
    print(b.groupby("method")[["whole_abs"]].median().join(
        b.assign(a=b.whole_signed.abs()).groupby("method").a.quantile(0.9).rename("p90_abs_signed")).round(1).to_string())
    print(nb.pivot(index="prog", columns="space", values="nearest").to_string())


if __name__ == "__main__":
    main()
