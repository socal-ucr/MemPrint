"""The cost of the CUDA interpreter against running the program on the GPU and tracing it with NVBit.

    python paper_gpu/overhead.py [--quick]

For each test program and input size: the program's own run time on the GPU (wall clock, median of three,
CUDA initialisation included), its run time under the NVBit tool (gpu/nvbit_rw/rw_trace.so), and the
interpreter's time under each thread schedule (interp_cuda.SCHEDULES; after lockstep, only the kernels that
read and wrote one unit in a launch are reordered), the part of it spent running kernels (the rest is the host
code), the number of events and of GPU threads it simulated, and the hybrid interpreter's time (host code
native) in lockstep and under the four schedules. Writes paper_gpu/data/overhead.csv. Needs a GPU (CUDA_VISIBLE_DEVICES), nvcc on
PATH and the NVBit tool built (gpu/README.md).
"""

import csv
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint.static import hybrid, interp_cuda as ic  # noqa: E402

CUDA = ROOT / "tests" / "static" / "cuda"
DATA = ROOT / "paper_gpu" / "data"
WORK = DATA / "overhead_runs"
TOOL = ROOT / "gpu" / "nvbit_rw" / "rw_trace.so"
LIMIT = 3600  # seconds an interpreter run may take before the larger sizes of that program are skipped

SIZES = {  # program -> input sizes (elements, or vertices of the graph)
    "chain": [10_000, 100_000, 1_000_000],
    "bfs": [2_000, 16_000, 128_000],
    "wl_bfs": [4_096, 16_384, 65_536],
    "rodinia_bfs": [4_096, 16_384, 65_536],
    "cc": [4_096, 16_384, 65_536],
    "ptrsel": [1_000, 10_000, 100_000],
}


def graph(n, path):
    """A random graph in Rodinia's format, 1 to 5 edges per vertex (scripts/kdeps_truth_tests.sh)."""
    random.seed(7)
    adj = [[random.randrange(n) for _ in range(random.randint(1, 5))] for _ in range(n)]
    starts = [0]
    for a in adj:
        starts.append(starts[-1] + len(a))
    lines = [str(n)] + [f"{starts[v]} {len(adj[v])}" for v in range(n)] + ["0", str(starts[-1])] + \
        [f"{u} 1" for a in adj for u in a]
    path.write_text("\n".join(lines) + "\n")


def argv_for(prog, n):
    if prog == "rodinia_bfs":
        g = WORK / f"graph_{n}.txt"
        if not g.exists():
            graph(n, g)
        return [str(g)]
    return [str(n)]


KERNEL_TIME = [0.0]
_run_kernel = ic.CudaInterpreter._run_kernel


def _timed_run_kernel(self, *a, **k):
    start = time.time()
    try:
        return _run_kernel(self, *a, **k)
    finally:
        KERNEL_TIME[0] += time.time() - start


ic.CudaInterpreter._run_kernel = _timed_run_kernel


def wall(cmd, env=None):
    start = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    return time.time() - start


def main():
    quick = "--quick" in sys.argv
    WORK.mkdir(parents=True, exist_ok=True)
    out = DATA / "overhead.csv"
    rows = []
    for prog, sizes in SIZES.items():
        exe = WORK / prog
        subprocess.run(["nvcc", "-O2", "-arch=native", "-o", str(exe), str(CUDA / f"{prog}.cu"),
                        str(ROOT / "gpu" / "probe.cpp")], check=True)
        so = hybrid.build([str(CUDA / f"{prog}.cu")], str(WORK / f"{prog}.so"))
        for n in sizes[:1] if quick else sizes:
            args = argv_for(prog, n)
            native = statistics.median(wall([str(exe)] + args) for _ in range(3))
            tdir = WORK / f"{prog}_{n}_truth"
            tdir.mkdir(exist_ok=True)
            env = dict(os.environ, LD_PRELOAD=str(TOOL), KDEPS_OUT=str(tdir), NOBANNER="1")
            try:
                traced = round(wall([str(exe)] + args, env), 3)
            except subprocess.CalledProcessError:                      # e.g. the GPU's memory taken by other users
                traced = float("nan")
            row = {"program": prog, "n": n, "native_s": round(native, 3), "nvbit_s": traced}
            slow = False
            racy = None                                    # as kdeps: other schedules reorder lockstep's racy kernels
            for schedule in ic.SCHEDULES:
                start, KERNEL_TIME[0] = time.time(), 0.0
                it, errors, _ = ic.analyze(CUDA / f"{prog}.cu", argv=[prog] + args, host_events=False,
                                           schedule=schedule, racy=racy)
                if schedule == "lockstep":
                    racy = set(it.racy_seen)
                row[f"interp_{schedule}_s"] = round(time.time() - start, 3)
                if schedule == "lockstep":
                    row["interp_kernels_s"] = round(KERNEL_TIME[0], 3)
                    row["events"] = len(it.tracker.events)
                    row["threads"] = sum(e["threads"] or 0 for e in it.tracker.events)
                    row["dependences"] = len(it.tracker.edges)
                slow |= row[f"interp_{schedule}_s"] > LIMIT
            # the hybrid interpreter: host code native, kernels interpreted (built once, outside the timing)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "analysis"))
            hdir = WORK / f"{prog}_{n}_hybrid"
            for label, sched in (("hybrid_lockstep_s", ["--schedule", "lockstep"]),
                                 ("hybrid_all_s", ["--schedules", ",".join(ic.SCHEDULES)])):
                start = time.time()
                subprocess.run([sys.executable, "-m", "memprint.static.hybrid", str(CUDA / f"{prog}.cu"),
                                str(hdir), "--name", prog, "--so", str(so), *sched, "--argv", prog, *args],
                               cwd=WORK, env=env, check=True, capture_output=True)
                row[label] = round(time.time() - start, 3)
            print(row, flush=True)
            rows.append(row)
            with open(out, "w", newline="") as f:
                wr = csv.DictWriter(f, list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            if slow:
                break


if __name__ == "__main__":
    main()
