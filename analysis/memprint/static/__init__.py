"""Static analysis of a workload's source: its access-count spectrum, the bin
statistics that spectrum implies, and access-idiom features for code the
interpreter cannot run."""

import time
from pathlib import Path

import numpy as np

from . import clangast, interp
from .spectrum import Spectrum

POLYBENCH_CONFIGS = ["MINI", "MINI2", "MINI3", "SMALL", "SMALL2", "SMALL3", "MEDIUM"]


def polybench_sources(root):
    """Kernel name -> source file, from utilities/benchmark_list."""
    root = Path(root)
    lines = (root / "utilities" / "benchmark_list").read_text().split()
    return {Path(line).stem: root / line for line in lines}


def analyze(source, defines=(), includes=(), cplusplus=False, footprint="starts"):
    """Run the interpreter on one program. Returns (Spectrum, interp.Result, parse errors, seconds).
    footprint: starts (largest access per start address) or bytes (bytes touched)."""
    start = time.time()
    tu = clangast.parse(source, defines, includes, cplusplus=cplusplus)
    errors = [str(d) for d in tu.diagnostics if d.severity >= 3]
    result = interp.run(tu, footprint=footprint)
    spec = Spectrum.from_addresses(result.counts, result.sizes, result.total)
    return spec, result, errors, time.time() - start


def analyze_polybench(root, kernel, config, footprint="starts"):
    root = Path(root)
    return analyze(polybench_sources(root)[kernel], [f"{config}_DATASET"], [root / "utilities"],
                   footprint=footprint)


def save(path, spec, result, errors, seconds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **spec.to_dict(), total=result.total, unresolved=result.unresolved,
                        uncertain=result.uncertain, data_loops=result.data_loops,
                        data_branches=result.data_branches, errors=len(errors), seconds=seconds)


def load(path):
    d = np.load(path)
    return Spectrum.from_dict(d), {k: float(d[k]) for k in ("total", "unresolved", "uncertain", "data_loops",
                                                              "data_branches", "errors", "seconds")}
