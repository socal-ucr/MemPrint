"""Second blind test: predictions for NPB IS, CG and MG (classes S, W, A) from their source, with the interpreter
frozen at d2258d8. python paper_generalization/blind2/predict.py [kernels] [classes]

Each class's sizes come from NPB's own setparams (npbparams.hpp); the kernel, NPB's common files and that header
are compiled as one unity file. The runtime baseline is fitted on the 12 serial GAP workloads (C++, -O3, as for
HPCCG and LULESH). Writes npb_<kernel>_predicted.csv here and the spectra to data/blind2/data/static_cpp.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint import irregular, similarity as sim  # noqa: E402
from memprint.dataset import prepare  # noqa: E402
from memprint.static import load, runs  # noqa: E402

NPB = ROOT / "workloads" / "src" / "npb" / "NPB-SER"
OUT = ROOT / "data" / "blind2" / "data"
GAP = ROOT / "data" / "gap-bytes" / "data"
COMMON = ["c_randdp.cpp", "c_timers.cpp", "c_print_results.cpp", "wtime.cpp"]
HEAP_TOP = 59328


def sources(kernel, cls):
    """kernel.cpp, NPB's common files and the class's npbparams.hpp, in a directory of their own."""
    d = OUT / "src" / f"{kernel}-{cls}"
    d.mkdir(parents=True, exist_ok=True)
    kdir = NPB / kernel.upper()                                        # setparams needs ../config/make.def
    subprocess.run([str(NPB / "sys" / "setparams"), kernel, cls], cwd=kdir, check=True, capture_output=True)
    shutil.copy(kdir / "npbparams.hpp", d / "npbparams.hpp")
    shutil.copy(NPB / kernel.upper() / f"{kernel}.cpp", d)
    common = d / "common"
    common.mkdir(exist_ok=True)
    for f in (NPB / "common").iterdir():
        if f.suffix in (".cpp", ".hpp", ".h"):
            shutil.copy(f, common)
    text = (d / f"{kernel}.cpp").read_text().replace('#include "../common/', '#include "')
    (d / f"{kernel}.cpp").write_text(text)
    return [d / f"{kernel}.cpp"] + [common / f for f in COMMON]


def baseline():
    srcs = []
    for w in ["gap_pr", "gap_bfs", "gap_cc", "gap_bc", "gap_tc", "gap_sssp", "gap_pr_kron", "gap_bfs_kron",
              "gap_cc_kron", "gap_bc_kron", "gap_tc_kron", "gap_sssp_kron"]:
        a = pd.read_csv(GAP / f"{w}_allData.csv").assign(Config=lambda d: d.Config.astype(str))
        specs = {f.stem.rsplit("-", 1)[1]: load(f)[0] for f in sorted((GAP / "static_auto").glob(f"{w}-*.npz"))}
        srcs.append((prepare(a), specs, sim.bin_summary(a)))
    return irregular.fit_baseline(srcs)


def main():
    kernels = sys.argv[1].split(",") if len(sys.argv) > 1 else ["is", "cg", "mg"]
    classes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["S", "W", "A"]
    base = baseline()
    for k in kernels:
        spectra, meta = {}, {}
        for c in classes:
            sp, me = runs.program_spectra(f"npb_{k}", [str(p) for p in sources(k, c)], [c], [],
                                          heap_top=HEAP_TOP, cache_dir=OUT / "static_cpp", n_jobs=1)
            spectra.update(sp)
            meta.update(me)
        pred = runs.predictions(f"npb_{k}", spectra, meta, base)
        pred.to_csv(HERE / f"npb_{k}_predicted.csv", index=False)
        print(pred[pred.k == 1][["config", "footprint", "program_footprint", "coverage", "seconds"]].to_string())


if __name__ == "__main__":
    main()
