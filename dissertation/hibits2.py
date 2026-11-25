"""Plan of the remaining sampler runs with the high-bit keep test (hibits2.sh): M3 at 20 bins (m20), M11 (m11), the
pages-touched sets p50 and cp50, and the corpus at 20 bins (c20), mirrored into $N/<set>/m3/<sampler>/<seed>/<w>/<c>/<k>.

A sampler run depends only on the sampler and its bin count, the seed, the program, its input and the interval (the
bins are in bytes under either truth), so each distinct run is made once: a run that an earlier mirror (b50, c50, m5,
m20) or an earlier set of this plan already has becomes a symbolic link to it.

    python3 dissertation/hibits2.py  ->  $N/jobs2.txt ("<GB> one2 <dir> <sampler> <bins> <seed> <w> <c> <k> <kind>"),
                                         the links of $N/<set>/m3/...
"""
import glob
import os
import re

H = os.path.expanduser("~/memprint-data")
N = f"{H}/2026-11-23-hibits"
O = f"{H}/2026-11-05-diss-ch2"
MORE = f"{H}/2026-11-12-b50more"
CM = f"{H}/2026-11-13-corpusmore"
CORPUS = {"gap-bfs", "gap-cc", "gap-pr", "gap-sssp", "hpccg", "lulesh", "minife"}
# set: (original root, disjoint bins: 20 means M3's rule, 25 when k is not a multiple of 20)
SETS = {"m20": (O, 20), "m11": (f"{O}/m11", 20), "p50": (f"{MORE}/p50", 50), "cp50": (f"{CM}/cp50", 50),
        "c20": (f"{CM}/c20", 20)}
DONE = {"b50": 50, "c50": 50, "m5": 50, "m20": 20}   # existing mirrors


def bins(rule, s, k):
    if s == "boot":
        return 0
    return 25 if rule == 20 and k % 20 else rule


def peaks():
    """GB per (workload, config): the largest peak of any earlier run of it x 1.3, at least 1."""
    kb = {}
    for line in open(f"{N}/done.txt"):
        x = line.split()
        if len(x) >= 10:
            kb[(x[3], x[4])] = max(kb.get((x[3], x[4]), 0), int(x[9]))
    for root, _ in SETS.values():
        for t in glob.glob(f"{root}/m3/*/*/*/*/*/time*"):
            w, c = t.split("/")[-4:-2]
            for l in open(t):
                m = re.match(r"\S+ (\d+)$", l.strip())
                if m:
                    kb[(w, c)] = max(kb.get((w, c), 0), int(m.group(1)))
    return kb


def main():
    have = {}
    for st, rule in DONE.items():
        for d in glob.glob(f"{N}/{st}/m3/*/*/*/*/*"):
            if os.path.islink(d):
                continue
            s, seed, w, c, k = d.split("/")[-5:]
            have.setdefault((s, bins(rule, s, int(k)), seed, w, c, k), d)
    kb = peaks()
    jobs, links = [], 0
    for st, (root, rule) in SETS.items():
        for d in sorted(glob.glob(f"{root}/m3/*/*/*/*/*")):
            s, seed, w, c, k = d.split("/")[-5:]
            target = f"{N}/{st}/m3/{s}/{seed}/{w}/{c}/{k}"
            if os.path.lexists(target):
                continue
            key = (s, bins(rule, s, int(k)), seed, w, c, k)
            if key in have:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.symlink(have[key], target)
                links += 1
                continue
            have[key] = target
            gb = max(1.0, kb.get((w, c), 3e6) / 1e6 * 1.3)
            kind = "corpus" if w in CORPUS else "poly"
            jobs.append(f"{gb:.1f} one2 {target} {s} {key[1]} {seed} {w} {c} {k} {kind}")
    open(f"{N}/jobs2.txt", "w").write("\n".join(jobs) + "\n")
    print(len(jobs), "runs,", links, "links")


if __name__ == "__main__":
    main()
