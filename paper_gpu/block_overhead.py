"""What dependences between blocks cost over dependences between operations, for the hybrid interpreter.

    [PAPER_GPU_SIZE=large] python paper_gpu/block_overhead.py [--jobs 8] [--only NAME,...] [--post-only]

Per benchmark of benchmarks.py with a hybrid run (data/bench*/NAME/hybrid): the interpreter in lockstep, without
and with per-block recording (hybrid.py --blocks), back to back on the same built program: wall time, peak memory
(/usr/bin/time) and the size of what it writes; then the time to refine its dependences to blocks from the four
schedules' references (simulate_blocks.hybrid_block) and the number of dependences either way.
Writes data/block_overhead{,_large}.csv.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmarks as B  # noqa: E402
import simulate as S  # noqa: E402
import simulate_blocks as SB  # noqa: E402

DATA = HERE / "data"


def interp(name):
    """Lockstep without and with --blocks: {ops,blocks}_{s,rss_mb,out_mb}."""
    b = B.BENCH[name]
    w = B.workdir(b)
    sd = B.source_dir(b, w) if not b.get("patches") else w / "src" / b.get("copy_as", ".")
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix=f"blkovh-{name}-"))
    try:
        for k, extra in (("ops", []), ("blocks", ["--blocks"])):
            out = tmp / k
            cmd = ["/usr/bin/time", "-f", "%e %M", B.PY, "-m", "memprint.static.hybrid",
                   ",".join(str(sd / s) for s in b["sources"]), str(out), "--name", name,
                   "--so", str(w / "hybrid" / f"{name}.so"),
                   *[x for d in B.COMPAT + b.get("defines", []) for x in ("-D", d)],
                   *[x for i in b.get("includes", []) for x in ("-I", str(sd / i))], *extra,
                   "--argv", name, *B.argv(b, w)]
            env = dict(os.environ, PYTHONPATH=str(B.ROOT / "analysis"))
            p = subprocess.run(cmd, cwd=w, env=env, capture_output=True, text=True, timeout=4 * 3600)
            t, rss = p.stderr.strip().splitlines()[-1].split()
            res[f"{k}_s"], res[f"{k}_rss_mb"] = float(t), round(int(rss) / 1024, 1)
            res[f"{k}_out_mb"] = round(sum(f.stat().st_size for f in out.iterdir() if f.suffix != ".so") / 1e6, 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def post(name):
    """Refining the four schedules' dependences to blocks: its time and the dependences either way."""
    w = B.workdir(B.BENCH[name])
    ev = list(csv.DictReader(open(w / "truth_blocks" / "truth_events.csv")))
    nb = [max(1, SB._grid_blocks(e["grid"])) if e["kind"] == "kernel" else 1 for e in ev]
    pairs = S._predicted(w / "hybrid", name, ev)
    t = time.time()
    whole, spec = SB.hybrid_block(name, w, ev, nb)
    return {"refine_s": round(time.time() - t, 2), "op_deps": len(pairs),
            "block_deps": sum(len(x) for per in spec.values() for x in per.values()),
            "whole_deps": sum(len(v) for v in whole.values()), "blocks": sum(nb)}


def run(name):
    r = {"name": name, "suite": B.BENCH[name]["suite"], **interp(name)}
    print(r, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--post-only", action="store_true", help="keep the interpreter columns of the existing file")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [
        n for n in B.BENCH if (B.workdir(B.BENCH[n]) / "hybrid" / "schedule_lockstep" / f"{n}_blocks.npz").exists()
        and (B.workdir(B.BENCH[n]) / "truth_blocks" / "truth_events.csv").exists()]
    path = DATA / f"block_overhead{B.SUFFIX}.csv"
    old = {r["name"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    if args.post_only:
        rows = [dict(old[n]) for n in names if n in old]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(run, names))
    for r in rows:                                                     # one at a time: timed alone
        r.update(post(r["name"]))
    hyb = {r["name"]: r for r in csv.DictReader(open(DATA / f"benchmarks{B.SUFFIX}.csv"))}
    for r in rows:
        r["hybrid_all_s"] = hyb.get(r["name"], {}).get("hybrid_s", "")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
