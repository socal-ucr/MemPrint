"""Benchmarks for paper_gpu: the whole-program interpreter and the hybrid interpreter on unmodified CUDA programs
from Rodinia, GraphBIG and PolyBench-GPU, against traced runs on the GPU.

    python paper_gpu/benchmarks.py [--gpu] [--interp] [--jobs 8] [--only NAME,...] [--timeout 3600]

--gpu   (GPU 1: CUDA_VISIBLE_DEVICES) builds each program, runs it plainly (median of three) and under the NVBit
        tool (gpu/nvbit_rw/rw_trace.so) into paper_gpu/data/bench/NAME/truth.
--interp runs, JOBS at a time, the whole-program interpreter (memprint.static.kdeps) and the hybrid interpreter
        (memprint.static.hybrid), each under the four thread schedules, into .../full and .../hybrid, and scores
        both against the trace (kdeps_score). Writes paper_gpu/data/benchmarks.csv.
Sources are workloads/src/{gpu-rodinia,graphBIG,polybenchGpu} (scripts/setup.sh), copied to the work
directory and patched there where CUDA 13 no longer has an API they use; inputs are generated small enough for
the tracer and the interpreters. Needs nvcc on PATH.

PAPER_GPU_SIZE=large selects larger inputs (LARGE below; GraphBIG on a generated graph of 8,192 vertices), with
the work directory paper_gpu/data/bench_large and paper_gpu/data/benchmarks_large.csv. There --gpu traces with
the block references kept (KDEPS_BLOCKS: truth_blocks is truth) and also counts each block's instructions
(KDEPS_WORK: truth_work); the small inputs get the latter with --work.
"""

import argparse
import csv
import io
import os
import random
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint.static import kdeps, kdeps_score as ks  # noqa: E402

SRC = ROOT / "workloads" / "src"
SIZE = os.environ.get("PAPER_GPU_SIZE", "small")
LARGE = SIZE == "large"
SUFFIX = "_large" if LARGE else ""
WORK = ROOT / "paper_gpu" / "data" / f"bench{SUFFIX}"
TOOL = ROOT / "gpu" / "nvbit_rw" / "rw_trace.so"
PY = sys.executable
SCHEDULES = "lockstep,blocks,reverse,random"
COMPAT = ["cudaThreadSynchronize=cudaDeviceSynchronize"]   # removed in CUDA 12; same meaning
TRACE_TIMEOUT = 3600  # seconds a run under the NVBit tool may take
COPIES = ("cudaMemcpy H2D", "cudaMemcpy D2H", "cudaMemcpy D2D", "cudaMemset")


# ---------------------------------------------------------------- inputs

def rodinia_graph(path, n=4096, seed=7):
    random.seed(seed)
    adj = [[random.randrange(n) for _ in range(random.randint(1, 5))] for _ in range(n)]
    starts = [0]
    for a in adj:
        starts.append(starts[-1] + len(a))
    lines = [str(n)] + [f"{starts[v]} {len(adj[v])}" for v in range(n)] + ["0", str(starts[-1])] + \
        [f"{u} 1" for a in adj for u in a]
    path.write_text("\n".join(lines) + "\n")


def graphbig_graph(out, n=8192, m=8, seed=11):
    """A GraphBIG dataset of n vertices by preferential attachment, m edges per new vertex (a skewed degree
    distribution like the social network's), both directions of every edge as in the small one: vertex.CSR and
    edge.CSR (uint64 offsets and targets, which the GPU benchmarks load) and the CSV files they came from."""
    if (out / "edge.CSR").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    ends, edges = list(range(m)), set()
    for v in range(m, n):
        for u in {random.choice(ends) for _ in range(m)}:
            edges.add((u, v))
            ends += [u, v]
    with open(out / "vertex.csv", "w") as f:
        f.write("id|firstName|lastName|gender|birthday|creationDate|locationIP|browserUsed\n")
        f.writelines(f"{v}|a|b|female|1980-01-01|2010-01-01T00:00:00Z|0.0.0.0|Firefox\n" for v in range(n))
    with open(out / "edge.csv", "w") as f:
        f.write("Person.id|Person.id\n")
        f.writelines(f"{a}|{b}\n{b}|{a}\n" for a, b in sorted(edges))
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    offsets = [0]
    for lst in adj:
        lst.sort()
        offsets.append(offsets[-1] + len(lst))
    (out / "vertex.CSR").write_bytes(struct.pack(f"<{n + 1}Q", *offsets))
    (out / "edge.CSR").write_bytes(struct.pack(f"<{offsets[-1]}Q", *(v for lst in adj for v in lst)))


def hotspot_files(work, n=64):
    random.seed(3)
    (work / "temp.txt").write_text("".join(f"{320 + random.random() * 20:.6f}\n" for _ in range(n * n)))
    (work / "power.txt").write_text("".join(f"{random.random() * 1e-3:.8f}\n" for _ in range(n * n)))


# ---------------------------------------------------------------- the benchmarks

def rodinia():
    r = SRC / "gpu-rodinia" / "cuda"
    out = [
        dict(name="rodinia-bfs", dir=r / "bfs", sources=["bfs.cu"], argv=["{work}/graph.txt"],
             prepare=lambda w: rodinia_graph(w / "graph.txt", n=65536 if LARGE else 4096)),
        dict(name="rodinia-gaussian", dir=r / "gaussian", sources=["gaussian.cu"],
             argv=["-s", "160" if LARGE else "64", "-q"],
             patches=[("gaussian.cu", r"(printf\(.*clockRate.*\);)", r"/* \1 */"),
                      ("gaussian.cu", r"(printf\(.*deviceOverlap.*\);)", r"/* \1 */")]),
        dict(name="rodinia-nw", dir=r / "nw", sources=["needle.cu"], argv=["1024" if LARGE else "128", "10"]),
        dict(name="rodinia-pathfinder", dir=r / "pathfinder", sources=["pathfinder.cu"],
             argv=["100000", "100", "20"] if LARGE else ["1000", "20", "5"]),
        dict(name="rodinia-hotspot", dir=r / "hotspot", sources=["hotspot.cu"],
             argv=["512" if LARGE else "64", "2", "2", "{work}/temp.txt", "{work}/power.txt", "{work}/hotspot.out"],
             prepare=lambda w: hotspot_files(w, 512 if LARGE else 64)),
        dict(name="rodinia-lud", dir=r / "lud", sources=["cuda/lud.cu", "cuda/lud_kernel.cu", "common/common.c"],
             includes=["common"], argv=["-s", "256" if LARGE else "64"]),
        dict(name="rodinia-backprop", dir=r / "backprop",
             sources=["backprop.c", "facetrain.c", "imagenet.c", "backprop_cuda.cu"],
             argv=["65536" if LARGE else "4096"]),
        dict(name="rodinia-srad", dir=r / "srad" / "srad_v2", sources=["srad.cu"],
             argv=["512" if LARGE else "64", "512" if LARGE else "64", "0", "15", "0", "15", "0.5", "2"]),
    ]
    for b in out:
        b["suite"] = "Rodinia"
    return out


def graphbig():
    g = SRC / "graphBIG" / "gpu_bench"
    data = SRC / "graphBIG" / "dataset" / "small"
    if LARGE:
        data = WORK / "graphbig-dataset"
        graphbig_graph(data)
    progs = [("gpu_BFS", "bfs_topo_thread_centric", True), ("gpu_BFS", "bfs_topo_atomic", True),
             ("gpu_BFS", "bfs_data_thread_centric", True), ("gpu_SSSP", "sssp_topo_thread_centric", True),
             ("gpu_ConnectedComp", "connected_comp", False), ("gpu_kCore", "kcore", False),
             ("gpu_GraphColoring", "gc_topo_thread_centric", False), ("gpu_DegreeCentr", "degree_centr", False),
             ("gpu_TriangleCount", "triangle_count", False), ("gpu_BetweennessCentr", "betweenness", False)]
    out = []
    for d, kern, root in progs:
        out.append(dict(name=f"graphbig-{kern.replace('_', '-')}", suite="GraphBIG", dir=g,
                        sources=[f"{d}/main.cpp", f"{d}/{kern}.cu"], includes=["../common", "../openG", "cudalib"],
                        defines=["NO_PFM"], flags=["-std=c++14"],
                        argv=["--dataset", str(data)] + (["--root", "31"] if root else [])))
    return out


def polybench():
    p = SRC / "polybenchGpu" / "CUDA"
    if LARGE:
        return polybench_sizes({
            "2DCONV": "NI=1024 NJ=1024", "2MM": "NI=256 NJ=256 NK=256 NL=256", "3DCONV": "NI=128 NJ=128 NK=128",
            "3MM": "NI=128 NJ=128 NK=128 NL=128 NM=128", "ATAX": "NX=4096 NY=4096", "BICG": "NX=4096 NY=4096",
            "CORR": "M=512 N=512", "COVAR": "M=512 N=512", "FDTD-2D": "TMAX=10 NX=384 NY=384",
            "GEMM": "NI=512 NJ=512 NK=512", "GESUMMV": "N=4096", "GRAMSCHM": "NI=256 NJ=256",
            "JACOBI1D": "TSTEPS=10 N=65536", "JACOBI2D": "TSTEPS=5 N=128", "LU": "N=256", "MVT": "N=4096",
            "SYR2K": "NI=512 NJ=512", "SYRK": "NI=512 NJ=512", "GEMVER": "N=1024", "ADI": "TSTEPS=1 N=256"})
    sizes = {"2DCONV": "NI=256 NJ=256", "2MM": "NI=128 NJ=128 NK=128 NL=128", "3DCONV": "NI=32 NJ=32 NK=32",
             "3MM": "NI=64 NJ=64 NK=64 NL=64 NM=64", "ATAX": "NX=512 NY=512", "BICG": "NX=512 NY=512",
             "CORR": "M=128 N=128", "COVAR": "M=128 N=128", "FDTD-2D": "TMAX=10 NX=256 NY=256",
             "GEMM": "NI=128 NJ=128 NK=128", "GESUMMV": "N=512", "GRAMSCHM": "NI=128 NJ=128",
             "JACOBI1D": "TSTEPS=10 N=1024", "JACOBI2D": "TSTEPS=5 N=128", "LU": "N=128", "MVT": "N=512",
             "SYR2K": "NI=128 NJ=128", "SYRK": "NI=128 NJ=128", "GEMVER": "N=512", "ADI": "TSTEPS=1 N=128"}
    return polybench_sizes(sizes)


def polybench_sizes(sizes):
    p = SRC / "polybenchGpu" / "CUDA"
    out = []
    for d, s in sizes.items():
        cu = sorted(x.name for x in (p / d).glob("*.cu"))
        cuh = sorted(x.name for x in (p / d).glob("*.cuh"))[0]
        # the sizes go into the header, in place of its default sizes (# if !defined(N) ... # endif): a -DN on the
        # command line would also rename parameters called N in cuda.h, which the programs include first
        defs = "".join(f"# define {x.split('=')[0]} {x.split('=')[1]}\n" for x in s.split())
        out.append(dict(name=f"polybench-{d.lower()}", suite="PolyBench-GPU", dir=p / d, sources=cu, argv=[],
                        copy_as=f"CUDA/{d}", links={"common": p.parent / "common"},
                        patches=[(cuh, r"# if !defined\(N\)", defs.replace("\\", "\\\\") + "# if 0")]))
    return out


BENCH = {b["name"]: b for b in rodinia() + graphbig() + polybench()
         if not (LARGE and b["name"] == "graphbig-betweenness")}   # ten times the next; timed out when small


# ---------------------------------------------------------------- one benchmark

def workdir(b):
    w = WORK / b["name"]
    w.mkdir(parents=True, exist_ok=True)
    return w


def source_dir(b, w):
    """The sources, patched in a copy under the work directory if the benchmark needs it."""
    if not b.get("patches"):
        return Path(b["dir"])
    top = w / "src"
    if top.exists():
        shutil.rmtree(top)
    dst = top / b.get("copy_as", ".")                      # the copy keeps the layout relative includes expect
    shutil.copytree(b["dir"], dst)
    for rel, target in b.get("links", {}).items():
        (top / rel).symlink_to(target)
    for f, pat, rep in b["patches"]:
        t = (dst / f).read_text(errors="replace")
        (dst / f).write_text(re.sub(pat, rep, t))
    return dst.resolve()


def argv(b, w):
    return [a.replace("{work}", str(w)) for a in b["argv"]]


def flags(b, sd):
    return [*(f"-D{d}" for d in COMPAT + b.get("defines", [])), *(f"-I{sd / i}" for i in b.get("includes", [])),
            *b.get("flags", [])]


def wall(cmd, cwd, env=None, timeout=None):
    """Run cmd; on a timeout, kill it and everything it started (the hybrid runs a process per schedule)."""
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return time.time() - start, subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def gpu(b):
    w = workdir(b)
    if b.get("prepare"):
        b["prepare"](w)
    sd = source_dir(b, w)
    exe = w / "native"
    res = {"name": b["name"]}
    done = subprocess.run(["nvcc", "-O2", "-arch=native", "-w", *flags(b, sd), "-o", str(exe),
                           *(str(sd / s) for s in b["sources"])], capture_output=True, text=True)
    if done.returncode:
        res["gpu_status"] = "build failed: " + done.stderr.strip().splitlines()[-1][:120]
        return res
    times = []
    for _ in range(3):
        t, done = wall([str(exe), *argv(b, w)], w)
        times.append(t)
    res["native_s"] = round(statistics.median(times), 3)
    res["native_rc"] = done.returncode
    (w / "native.log").write_text(done.stdout + done.stderr)
    truth = w / "truth"
    truth.mkdir(exist_ok=True)
    env = dict(os.environ, LD_PRELOAD=str(TOOL), KDEPS_OUT=str(truth), NOBANNER="1")
    if LARGE:                                                          # the block references too (simulate_blocks)
        env["KDEPS_BLOCKS"] = "1"
        if not (w / "truth_blocks").exists():
            (w / "truth_blocks").symlink_to("truth")
    try:
        t, done = wall([str(exe), *argv(b, w)], w, env, TRACE_TIMEOUT)
    except subprocess.TimeoutExpired:
        res["gpu_status"] = "trace timed out"
        return res
    (w / "traced.log").write_text(done.stdout + done.stderr)
    res["nvbit_s"] = round(t, 3)
    res["gpu_status"] = "ok" if (truth / "truth_events.csv").exists() else "trace failed"
    if LARGE and res["gpu_status"] == "ok":
        res.update(work(b))
    return res


def _ops(path):
    with open(path) as f:
        return [(r["kind"], r["name"]) for r in csv.DictReader(f)]


def work(b, tries=8):
    """Count each block's instructions (KDEPS_WORK) into truth_work, in a run that issues the operations of the
    block trace (truth_blocks)."""
    w = workdir(b)
    out = w / "truth_work"
    out.mkdir(exist_ok=True)
    want = _ops(w / "truth_blocks" / "truth_events.csv")
    env = dict(os.environ, LD_PRELOAD=str(TOOL), KDEPS_OUT=str(out), NOBANNER="1", KDEPS_WORK="1")
    for k in range(tries):
        try:
            t, done = wall([str(w / "native"), *argv(b, w)], w, env, TRACE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"work_status": "timed out"}
        if (out / "truth_events.csv").exists() and _ops(out / "truth_events.csv") == want:
            return {"work_s": round(t, 3), "work_status": f"ok (try {k + 1})"}
    return {"work_status": "no run matched the block trace"}


def interp(b, which, timeout):
    """Run one interpreter on one benchmark: (seconds, status, scores)."""
    w = workdir(b)
    sd = source_dir(b, w) if not b.get("patches") else w / "src" / b.get("copy_as", ".")
    out = w / which
    srcs = ",".join(str(sd / s) for s in b["sources"])
    dflags = [x for d in COMPAT + b.get("defines", []) for x in ("-D", d)]
    iflags = [x for i in b.get("includes", []) for x in ("-I", str(sd / i))]
    mod = "memprint.static.kdeps" if which == "full" else "memprint.static.hybrid"
    cmd = [PY, "-m", mod, srcs, str(out), "--name", b["name"], *dflags, *iflags, "--access",
           "--schedules", SCHEDULES]
    if which == "full":
        cmd.append("--no-host")
    else:
        cmd += ["--blocks", "--device-layout"]                        # per-block references (simulate_blocks.py);
                                                                       # out-of-bounds references where the GPU puts them
    cmd += ["--argv", b["name"], *argv(b, w)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "analysis"))       # run in the work directory: files the
    if b.get("flags"):                                                 # program writes stay there
        env["NVCC_APPEND_FLAGS"] = " ".join(b["flags"])            # nvcc reads it: the hybrid's build of the program
    try:
        t, done = wall(cmd, w, env, timeout)
    except subprocess.TimeoutExpired:
        return {f"{which}_s": timeout, f"{which}_status": "timeout"}
    (w / f"{which}.log").write_text(done.stdout + done.stderr)
    res = {f"{which}_s": round(t, 2)}
    errs = [l for l in done.stderr.splitlines() if l.startswith("parse error")]
    res[f"{which}_parse_errors"] = len(errs)
    res[f"{which}_host_warning"] = int("not fully understood" in done.stderr)
    res[f"{which}_out_of_bounds"] = int("past the end of a device buffer" in done.stdout + done.stderr)
    if not (out / f"{b['name']}_events.csv").exists():
        last = (done.stderr.strip().splitlines() or ["no output"])[-1]
        res[f"{which}_status"] = "failed: " + last[:150]
        return res
    res[f"{which}_status"] = "ok" if done.returncode == 0 else f"ran with errors (exit {done.returncode})"
    truth = w / "truth"
    if (truth / "truth_events.csv").exists():
        res.update({f"{which}_{k}": v for k, v in score(out, b["name"], truth).items()})
    return res


def score(out, name, truth):
    ie, _ = ks.interp_edges(out, name)
    te, _ = ks.truth_edges(truth, 4)
    scores, _, tm, _ = ks.score_run(out, name, truth, 4)
    row = {"aligned": scores.pop("aligned"), "ops": len([e for e in ie if e["kind"] != "host"]), "truth_ops": len(te)}
    row.update(scores)
    # dependences between two kernels through data-dependent bytes, and how many the trace has
    rows = list(csv.DictReader(open(out / f"{name}_edges.csv")))
    alloc = {r["name"]: int(r["alloc"]) for r in csv.DictReader(open(out / f"{name}_objects.csv")) if r["alloc"]}
    kk = [r for r in rows if r["src_name"] not in COPIES and r["dst_name"] not in COPIES and r["object"] in alloc]
    dd = [r for r in kk if int(r["units_dd"]) > 0]
    row["kernel_deps"], row["data_deps"] = len(kk), len(dd)
    row["data_must"] = sum(r["certainty"] == "must" for r in dd)
    row["data_must_traced"] = sum(r["certainty"] == "must" and (int(r["src"]), int(r["dst"]), r["kind"],
                                                                 alloc[r["object"]]) in tm for r in dd)
    return row


def run_interp(args):
    name, which, timeout = args
    try:
        return name, interp(BENCH[name], which, timeout)
    except Exception as e:                                         # one benchmark's failure ends only its row
        return name, {f"{which}_status": f"driver error: {type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--interp", action="store_true")
    ap.add_argument("--work", action="store_true", help="count each block's instructions (truth_work)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--which", default="hybrid,full", help="the interpreters to run (--interp)")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(BENCH)
    WORK.mkdir(parents=True, exist_ok=True)
    summary = ROOT / "paper_gpu" / "data" / f"benchmarks{SUFFIX}.csv"
    rows = {}
    if summary.exists():
        rows = {r["name"]: r for r in csv.DictReader(open(summary))}
    for n in names:
        rows.setdefault(n, {"name": n})
        rows[n]["suite"] = BENCH[n]["suite"]

    def save():
        cols = []
        for r in rows.values():
            cols += [k for k in r if k not in cols]
        with open(summary, "w", newline="") as f:
            wr = csv.DictWriter(f, cols)
            wr.writeheader()
            wr.writerows(rows[n] for n in BENCH if n in rows)

    if args.gpu:
        for n in names:
            r = gpu(BENCH[n])
            print(r, flush=True)
            rows[n].update(r)
            save()
    if args.work:
        for n in names:
            r = work(BENCH[n])
            print(n, r, flush=True)
            rows[n].update(r)
            save()
    if args.interp:
        jobs = [(n, which, args.timeout) for n in names for which in args.which.split(",")]
        with ProcessPoolExecutor(args.jobs) as pool:
            for n, r in pool.map(run_interp, jobs):
                print(n, r, flush=True)
                rows[n].update(r)
                save()


if __name__ == "__main__":
    main()
