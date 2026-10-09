# MemPrint

MemPrint is a PIN-based tool to estimate program memory footprint from sparsely sampled memory traces.

A Pin tool records the footprint of a program at several sampling intervals. A log-linear model learns, from small inputs, how the footprint observed in a subsample relates to the true footprint. Given a cheap sampled run on a new input, the model then predicts that input's true footprint.

```
pintool/     memprint_trace Pin tool (splitter and sampler modes)
workloads/   one directory per workload: upstream commit, patches, build/run hooks
scripts/     setup.sh (fetch + patch workloads), run.sh (trace and time runs)
analysis/    Python package: traces -> training data -> models -> figures;
             static/ predicts footprint and alpha from source code
tests/       pintool/ (live-footprint checks), static/ (interpreter tests and C graph programs)
paper/                 report: footprint over time (reference, spatial and windowed sampling)
paper_generalization/  report: predicting models for unseen programs from source code
FINDINGS.md  every experiment, its results and how to reproduce it
```

## Requirements

- Linux x86-64, gcc, git, perl, bc
- Intel Pin 3.30 (`pin-3.30-98830-g1d7b601b3-gcc-linux`), from Intel's Pin download page
- Python 3.10+ with `analysis/requirements.txt`
- Optional: valgrind (massif baseline), an MPI C++ compiler for miniVite
  (e.g. `module load mpi/openmpi-x86_64`)

## Quick start

```bash
export PIN_ROOT=/path/to/pin-3.30-98830-g1d7b601b3-gcc-linux

# 1. Build the Pin tool
make -C pintool PIN_ROOT=$PIN_ROOT

# 2. Fetch and patch the workloads (into workloads/src/)
scripts/setup.sh all

# 3. Training traces: full trace split into subsamples, plus massif baseline
scripts/run.sh polybench --mode splitter --massif
scripts/run.sh minivite  --mode splitter --massif

# 4. Analysis (from the repository root)
pip install -r analysis/requirements.txt
export PYTHONPATH=$PWD/analysis
python -m memprint preprocess            # traces/<wl>/ -> data/<wl>_allData.csv
python -m memprint build --latex         # data/models.csv, data/fits.csv, accuracy table rows
python -m memprint predict               # data/{extrapolation,interpolation}_errors.csv
python -m memprint transform             # data/transformation_matrix.csv
python -m memprint paper-figures         # figures/paper/*.pdf
```

`scripts/config.sh` sets the paths. Each can be overridden from the environment: `PIN_ROOT`, `MEMPRINT_TOOL`, `WORKLOAD_SRC_DIR`, `TRACE_DIR` (default `traces/`) and `RESULTS_DIR` (default `results/`).

## Pin tool

```
pin -t pintool/obj-intel64/memprint_trace.so -mode splitter|sampler|spatial [knobs] -- <program> <args>
```

| Knob | Default | Meaning |
|---|---|---|
| `-mode` | `splitter` | `splitter`: trace every memory reference; write the exact footprint and, for each interval in `-intervals`, `-bins` disjoint bins each holding a 1-in-interval subsample. `sampler`: sample 1-in-`i` references and Poisson-bootstrap them into `-s` bins. `spatial`: select 1-in-`i` 8-byte words (start addresses with `-footprint starts`) by a salted hash and record every access to them, split into `-s` hash buckets. |
| `-intervals` | `100,250,...,100000` | splitter sampling intervals (13 by default) |
| `-bins` | 20 | splitter bins per interval |
| `-i` | 100 | sampler interval (`4294967295` = instrumentation only); spatial selection interval |
| `-s`, `-r` | 100, 10 | sampler bins and replication; λ = s/r, so each bin is about a 1-in-(i·r) subsample |
| `-outdir` | `traces` | output directory (created if missing) |
| `-name` | binary name | trace name, `<workload>-<config>` (run.sh sets it) |
| `-seed` | process id | RNG seed base |
| `-track_frees` | 0 | remove freed ranges (malloc family, munmap) from every footprint; see "Footprint over time" |
| `-snapshot` | 0 | append every footprint to `*_timeline.csv` every N memory references |
| `-stop` | 0 | write the outputs after N references and detach |
| `-window`, `-period` | 0 | spatial: watch accesses only for W of every P references, tracking allocations and page residency in between (needs `-snapshot`) |
| `-window_alloc` | 1048576 | windowed: an allocation of at least N bytes opens a window (0: off) |
| `-dirty_every` | 10 snapshots | windowed: clear soft-dirty bits every N references (0: residency only) |
| `-footprint` | see text | `bytes`: bytes touched, the union of every access's byte range. `starts`: the paper's definition, the sum of the largest access size at each start address. Default: `bytes` with `-track_frees` or `-window`, else `starts` |

The tool writes one CSV per footprint: `Buffered_<name>_1_<pid>.csv` (splitter, exact) or `Sampled_<name>_<i>_<pid>.csv` (sampler), plus `..._SubSample_<interval>_bin_<j>.csv` for every bin. Each file contains a single row:

```
FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval
Total,<bytes>,<unique addresses>,<references>,<interval>
```

## Running workloads

```
scripts/run.sh <workload> --mode splitter|sampler|spatial|instr|native
               [--bench "2mm gemm"] [--configs "MINI SMALL"] [--runs 5] [--pin-runs N]
               [--massif] [-i 1000 -s 100 -r 10] [--intervals LIST] [--bins N]
               [--footprint bytes|starts] [--snapshot N] [--track-frees] [--stop N]
               [--window W --period P]
```

The options after `--bins` pass the Pin knobs of the same name (`--track-frees` sets `-track_frees 1`).

- Traces go to `traces/<bench>/`.
- Timings go to `results/<bench>/overhead_<mode>.csv`, with these columns: native, Pin and massif times and overheads, averaged over the runs.
- Massif peaks go to `results/<bench>/massif.csv`.
- `instr` mode runs the sampler with an interval so large that it never samples, which measures the cost of instrumentation alone.

| Workload | Benchmarks | Configs |
|---|---|---|
| `polybench` | the 30 PolyBench/C 4.2.1 kernels | MINI, MINI2, MINI3, SMALL, SMALL2, SMALL3, MEDIUM |
| `minivite` | miniVite | 1024 … 16384 vertices (`-n`) |
| `gapbs` | GAP (b5e3e19, `-O3`): `gap_<k>` (uniform graph, `-u`) and `gap_<k>_kron` (Kronecker, `-g`) for k = bfs, pr (20 iterations), prc (pr to convergence), cc, sssp, tc, bc; `gap_{bfs,pr}[_kron]_t4` with 4 OpenMP threads | scale 10 … 16 by default (2^scale vertices, degree 16); the results use 10 … 18 |

### Adding a workload

Copy `workloads/TEMPLATE/` to `workloads/<name>/` and fill in `workload.sh`:

- `UPSTREAM_URL` and `UPSTREAM_COMMIT`
- `DEFAULT_CONFIGS`
- `wl_benchmarks`, `wl_build`, `wl_binary`
- optional `wl_setup`, `wl_args`, `wl_name`

Put changes against the upstream sources in `patches/*.patch`. `scripts/setup.sh <name>` and `scripts/run.sh <name>` need no other changes. The analysis splits trace names at the last `-` into workload and config, and orders numeric configs by value.

## Analysis

`python -m memprint [--root DIR] <command>` reads `traces/` and `results/` under `--root` (the current directory by default) and writes `data/` and `figures/` there. The workloads modelled are listed in `analysis/workloads.txt`.

- `preprocess [wl...]`: reads the splitter traces and writes `data/<wl>_allData.csv`, with the spread (SD) across bins for every config and interval.
- `build [wl...] [--latex]`: prepares the training data in four steps:
  1. Drop intervals 1 and 10, and any interval with an empty bin.
  2. Order configs by true footprint.
  3. Compute Alpha = true / observed footprint.
  4. Score each interval by how monotonically its spread grows with input size (Spearman).

  It then fits a model for each training subset (NZ: all intervals; MT: monotone intervals; L2O: the best monotone interval with one interval below it and three above) and each held-out config (EXTRA: largest; INTER: middle). All fits go to `data/fits.csv`. For each workload and split, the selected model (NZ or L2O, evaluated at the best monotone interval) goes to `data/models.csv`. `--latex` prints the rows of the paper's accuracy tables.
- `predict [--coef-decimals 4]`: applies every workload's model to every workload's data and writes the error matrices `data/{extrapolation,interpolation}_errors.csv` (rows: data, columns: model).
- `transform`: builds an RBF similarity over the model coefficients and computes the transformation T = pinv(S)·E onto the extrapolation errors.
- `plot sd-config|alpha|massif|heatmaps|paper ...`: draws a single figure type. See `--help`.

### Paper figures

`python -m memprint paper-figures` writes each data figure of the paper to `figures/paper/`, using the paper's file name:

| Figure | Command |
|---|---|
| `{gemm,miniVite}_baseline_vs_massif.pdf` | `plot massif gemm miniVite` |
| `{gemm,miniVite}_sd_vs_config.pdf` | `plot sd-config gemm miniVite` |
| `{gemm,miniVite}_config_vs_alpha_title.pdf` | `plot alpha gemm miniVite --x config` |
| `{gemm,miniVite}_rate_vs_alpha_title.pdf` | `plot alpha gemm miniVite --x rate` |
| `{gemver,gemm,floyd-warshall,miniVite}_sd_vs_config_highlight.pdf` | `plot sd-config ... --highlight best` |
| `miniVite_PIN_overhead.pdf`, `pin_splitter_overhead.pdf`, `cumulative_overhead_vs_accuracy.pdf`, `memory_usage_comparison.pdf` | `plot paper` (measured values in `analysis/memprint/plots/paper.py`) |

`SPLITTER.pdf` and `sampler.png` are diagrams and are not generated.

When run on the original traces and data, all 16 figures come out pixel-identical to the published ones, and `build --latex` reproduces every cell of the training and test accuracy tables.

## Unseen workloads from source (experimental)

`python -m memprint static ...` predicts a workload's footprint and α from its source, with no traces of it. It runs the source through an abstract interpreter (`analysis/memprint/static/`) that counts the references made to each address (the access-count spectrum). Under the splitter's Bernoulli sampling, the bins' mean footprint, spread and unique addresses at every interval then follow in closed form (`static/spectrum.py`). A small runtime baseline (loader, libc, allocator), fitted by NNLS on other workloads' traces, is added.

- **C** (`static/interp.py`): counts what an `-O0` build references. It runs counted loops as numpy batches and tracks integers exactly, including integer values stored in memory. It draws `rand()` from its distribution and places heap blocks as glibc does. Coverage is the share of references it placed exactly.
- **C++** (`static/interp_cpp.py`, `static/containers.py`): counts as `-O3` does, keeping small objects in registers and eliminating redundant loads. It models objects, references, templates, lambdas, `std::vector` / `unordered_map` / `pair` and the standard-library calls GAP uses. It runs GAP's unmodified source for all seven kernels on both graph families.

The commands:

| Command | What it does | Output |
|---|---|---|
| `static spectra --polybench DIR [--footprint bytes\|starts]` | computes each PolyBench kernel's spectrum at every config, for the footprint definition of the traces it is compared with (default bytes touched) | `data/static/<wl>-<config>.npz` |
| `static idioms --polybench DIR [--program name=root:files]` | classifies the access idioms of code the interpreter cannot run, and decides whether the static route applies | `data/static_idioms.csv` |
| `static lowo [--ast CSV] [--transfer miniVite]` | evaluates leave one workload out: each kernel is predicted from its source and the other kernels' traces | `data/lowo_*.csv`, `figures/static/` |
| `static gap [--borrow DIR]` | GAP pilot: predicts every GAP workload from a hand-written skeleton of its code (`static/gap.py`, `static/gap_kernels.py`) and the input graph's distribution, and scores it against its own model and borrowed PolyBench models | `data/gap_pilot_*.csv` |
| `static programs [--borrow DIR]` | runs the C interpreter on the irregular C programs `tests/static/csr_{pr,bfs}.c` and scores them | `data/programs_*.csv` |

The C++ interpreter has no command yet. Call it from Python; command-line accessors come from a table of stub values:

```python
from memprint.static import interp_cpp
from memprint.static.skeleton import HEAP_TOP_AT_START

cli = {"scale": 8, "degree": 16, "uniform": 1, "symmetrize": 1, "in_place": 0,
       "filename": interp_cpp.Str(""), "num_trials": 1, "max_iters": 20, "tolerance": 0,
       "logging_en": 0, "do_analysis": 0, "do_verify": 0, "start_vertex": -1,
       "ParseArgs": 1, "num_iters": 1, "delta": 1}
spec, result, errors, seconds = interp_cpp.analyze(
    "workloads/src/gapbs/src/pr.cc", heap_top=HEAP_TOP_AT_START, stubs={"__CL__": cli})
print(spec.footprint, result.coverage)    # 75845.0 1.0, in about 3 s
```

A GAP program takes about 10 s at scale 10, doubling with every scale (about 70 minutes and 4–5 GB at scale 18). `python tests/static/test_cpp.py` checks the C++ features on `tests/static/cpp_features.cc`.

Trace with `scripts/run.sh ... --footprint bytes` to compare against bytes touched. The parser is libclang (`pip install libclang`), so no clang binary is needed.

**Results** (FINDINGS.md, "Unseen workloads from source code alone" and "GAP pilot"):
- **PolyBench**, each of 27 kernels held out in turn:
  - footprint within a median of 0.09% (largest input held out) / 0.27% (middle input);
  - α within 1.2% / 5.5%. The kernel's own model gets 18.0% / 10.9%; the best borrowed model, chosen after the fact, 8.9% / 6.1%.
- **Choosing a model to borrow:** a descriptor predicted from source ranks which known model transfers best (median Spearman ρ 0.83–0.85), as well as the measured descriptor.
- **GAP pr and bfs from their C++ source**, scales 13–18: α within 0.8–4.7%, footprint within 0.44%. tc and bc are less accurate than the hand skeleton (12–18% against about 9% and 5%).

`paper_generalization/` is a detailed report of this work. To regenerate its data, figures and tables from `data/`:

```bash
python paper_generalization/compute.py <dir of interpreter *.time files>
python paper_generalization/make_figures.py
python paper_generalization/make_tables.py
cd paper_generalization && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Footprint over time (experimental)

The Pin tool can also record how the footprint evolves:

```bash
scripts/run.sh polybench --bench 2mm --mode splitter --snapshot 100000 --track-frees
scripts/run.sh polybench --bench 2mm --mode sampler -i 50 -s 20 -r 20 --snapshot 100000 --track-frees
python -m memprint timeline preprocess 2mm     # traces/2mm/*_timeline.csv -> data/2mm_timeline.csv
python -m memprint timeline build 2mm          # evaluation + data/timeline_models.csv
python -m memprint plot timeline 2mm
python -m memprint timeline estimate 2mm --run <dir with Sampled_*_timeline.csv>
python -m memprint timeline forecast 2mm --run <dir> --upto <references>
```

- **`-snapshot N`:** every N memory references, the tool appends the exact (splitter) or union (sampler) footprint, and every bin's footprint, to `*_timeline.csv`.
  - Time counts memory references executed. The sampler estimates it as sampled references × `-i`, which comes within 0.4% of the true count on jacobi-2d SMALL.
- **`-track_frees 1`:** `free`, `realloc` (the moved block or the shrunk tail) and `munmap` remove the released range from every footprint, so the footprint is live memory.
  - Releases are applied in program order relative to the buffered accesses of the same thread.
  - `free` is handled at its entry, because glibc's `free` exits through a tail jump.
  - A freed block releases its whole glibc chunk, read from the chunk's size field at `free`'s entry: the block, its size field and the slack after the requested size, which string and copy functions read past the end of a block. If the size field does not fit a chunk of the block's size (an mmapped chunk, another allocator), only the requested size is released.
  - Accesses made by the allocator's own code (glibc's `malloc.c`: chunk headers, free-list links) count as time but not toward any footprint, since they land outside live blocks and would never be released. Without this, a program that allocates and frees 600,000 small blocks keeps 5.7 MB of them "live". The code is found by symbol name; if libc has no symbol table, only its public allocation functions are covered.
  - Not tracked: `mremap`, `brk`, and stack frames. Ordering across threads is the order in which their buffers are processed.
  - The page index this needs roughly doubles the tool's memory use. miniVite 8192 uses about 14 GB.
- **Footprint definition.** With `-track_frees` or `-window`, footprints count bytes touched (`-footprint bytes`). The paper's definition (`-footprint starts`, the default otherwise) adds up the largest access at each start address, so accesses that overlap count more than once: an 8-byte read at every byte offset of a 1 MB buffer gives 8.5 MB, against 1.15 MB of bytes touched. Spatial and windowed sampling in bytes mode select aligned 8-byte words rather than start addresses, and keep the bytes of each access inside selected words, so selected bytes × i is unbiased.
- **Overhead** on 2mm SMALL:
  - splitter: 1.48 s plain, 1.61 s with `-snapshot 32000` (200 snapshots), 2.17 s adding `-track_frees` (bytes touched);
  - sampler: 1.47 s plain, 1.57 s with both.
- **`-stop N`:** write all outputs after N references and detach, which gives a partial trace.
- `tests/pintool/run_tests.sh` checks the live footprint over time for malloc/free, posix_memalign, realloc (shrink in place and move), calloc and new/delete, mmap with partial munmap, cross-thread frees, and `-stop`.

`timeline build` holds out the largest and the middle config, as `build` does. It evaluates two things:
- **Reconstruction:** the paper's α model applied at every snapshot, fitted on all snapshots of the training configs, to both the held-out config's splitter bins and a real sampler run.
- **Forecasting:** from 10–75% prefixes of the run, by fitting scaled versions of the training configs' curves.

**Results.** FINDINGS.md has the full results and `paper/memprint_timeline.tex` a detailed report. In short, with bytes touched and full traces as the truth:
- **Reference sampling** (α model per snapshot, Chao1/iChao1, a known-rate estimator and a hybrid with a learned correction) does not reach low error at sparse rates: the best, the hybrid, has 4–6% mean error at 1-in-3 references but 24% at 1-in-25 on the largest PolyBench size. A sample of references cannot tell new memory from memory touched again.
- **Spatial sampling** (`-mode spatial`) needs no training: 1.6–4.9% mean error on PolyBench at 1-in-100 words or denser, 0.3–6.8% on miniVite with 1, 4 and 16 threads. It still tests every access.
- **Windowed sampling** (`-window`, `-period`) tests accesses only in windows and tracks allocations, resident pages and (for reused heap blocks) soft-dirty written pages all the time. At 5% watched the peak is within 0.1% on PolyBench LARGE, within +1.8% on miniVite, and within −3.7% to +6.7% on GAP, darknet, Python, Lua, Perl, SQLite and GCC's cc1. `python -m memprint timeline windowed <wl>` scores such runs.
- The true peaks never exceed the peak RSS, and the tool's tracked allocations agree with Valgrind's Massif and DHAT where both see the same memory.

## Differences from the original tooling

This repository replaces the scripts in `memory_estimator` (Pin `ManualExamples/buffer_memtrace_*`, `workloads/*/run*.sh`, `tools/*.py`). Before the fixes below were applied, the refactored code was checked against the original on the same inputs: Pin outputs were byte-identical with a fixed seed, and all analysis outputs matched. The fixes change results:

- **Pin tool**
  - Addresses are now kept as 64 bits. They used to be truncated to 32 bits.
  - Splitter subsampling rates are now exact. Integer division used to turn 250 and 750 into 240 and 740. A modulo test on the LCG's low bits also biased some intervals by up to 11%.
  - Sampler bin files are now labelled with their real interval, about i·r. They used to be labelled i·s/r, which was 2× too high for `-s 50 -r 5`.
- **PolyBench**
  - `heat-3d.h` guarded its MINI2/MINI3 sizes with `MINI_DATASET`, so MINI was built with the MINI3 sizes and MINI2/MINI3 did not compile. This is now fixed, so heat-3d MINI is smaller than in the published data.
  - Kernels are still built without optimisation flags (empty `config.mk`), as they were for the published traces.
- **Cross-workload errors**
  - `predict` now uses full-precision coefficients.
  - The original pipeline read 4-decimal coefficients, which moves many matrix entries by several percentage points.
  - `--coef-decimals 4` reproduces the original matrices exactly.
