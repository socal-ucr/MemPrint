# MemPrint

MemPrint is a PIN-based tool to estimate program memory footprint from sparsely sampled memory traces.

A Pin tool records the footprint of a program at several sampling intervals. A log-linear model learns, from small inputs, how the footprint observed in a subsample relates to the true footprint. Given a cheap sampled run on a new input, the model then predicts that input's true footprint.

```
pintool/     memprint_trace Pin tool (splitter and sampler modes)
workloads/   one directory per workload: upstream commit, patches, build/run hooks
scripts/     setup.sh (fetch + patch workloads), run.sh (trace and time runs)
analysis/    Python package: traces -> training data -> models -> figures
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
pin -t pintool/obj-intel64/memprint_trace.so -mode splitter|sampler [knobs] -- <program> <args>
```

| Knob | Default | Meaning |
|---|---|---|
| `-mode` | `splitter` | `splitter`: trace every memory reference; write the exact footprint and, for each interval in `-intervals`, `-bins` disjoint bins each holding a 1-in-interval subsample. `sampler`: sample 1-in-`i` references and Poisson-bootstrap them into `-s` bins. |
| `-intervals` | `100,250,...,100000` | splitter sampling intervals (13 by default) |
| `-bins` | 20 | splitter bins per interval |
| `-i` | 100 | sampler interval (`4294967295` = instrumentation only) |
| `-s`, `-r` | 100, 10 | sampler bins and replication; λ = s/r, so each bin is about a 1-in-(i·r) subsample |
| `-outdir` | `traces` | output directory (created if missing) |
| `-name` | binary name | trace name, `<workload>-<config>` (run.sh sets it) |
| `-seed` | process id | RNG seed base |

The tool writes one CSV per footprint: `Buffered_<name>_1_<pid>.csv` (splitter, exact) or `Sampled_<name>_<i>_<pid>.csv` (sampler), plus `..._SubSample_<interval>_bin_<j>.csv` for every bin. Each file contains a single row:

```
FunctionName,MemUsageObs,UniqueAddresses,CountObs,SamplingInterval
Total,<bytes>,<unique addresses>,<references>,<interval>
```

## Running workloads

```
scripts/run.sh <workload> --mode splitter|sampler|instr|native
               [--bench "2mm gemm"] [--configs "MINI SMALL"] [--runs 5] [--pin-runs N]
               [--massif] [-i 1000 -s 100 -r 10] [--intervals LIST] [--bins N]
```

- Traces go to `traces/<bench>/`.
- Timings go to `results/<bench>/overhead_<mode>.csv`, with these columns: native, Pin and massif times and overheads, averaged over the runs.
- Massif peaks go to `results/<bench>/massif.csv`.
- `instr` mode runs the sampler with an interval so large that it never samples, which measures the cost of instrumentation alone.

| Workload | Benchmarks | Configs |
|---|---|---|
| `polybench` | the 30 PolyBench/C 4.2.1 kernels | MINI, MINI2, MINI3, SMALL, SMALL2, SMALL3, MEDIUM |
| `minivite` | miniVite | 1024 … 16384 vertices (`-n`) |

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
  - `free` is handled at its entry, because glibc's `free` exits through a tail jump. The 16 bytes of tcache links that free writes into a small block are counted again.
  - Not tracked: `mremap`, `brk`, and stack frames. Ordering across threads is the order in which their buffers are processed.
  - The page index this needs roughly doubles the tool's memory use. miniVite 8192 uses about 14 GB.
- **`-stop N`:** write all outputs after N references and detach, which gives a partial trace.
- `tests/pintool/run_tests.sh` checks the live footprint over time for malloc/free, posix_memalign, realloc (shrink in place and move), calloc and new/delete, mmap with partial munmap, cross-thread frees, and `-stop`.

`timeline build` holds out the largest and the middle config, as `build` does. It evaluates two things:
- **Reconstruction:** the paper's α model applied at every snapshot, fitted on all snapshots of the training configs, to both the held-out config's splitter bins and a real sampler run.
- **Forecasting:** from 10–75% prefixes of the run, by fitting scaled versions of the training configs' curves.

**Current results:** on 2mm, gemm, jacobi-2d and atax, the per-snapshot reconstruction error is around 30–45% MAPE. Forecasts of run length and peak are unreliable for extrapolation.

The cause is fundamental, not a tuning problem. A single bin is so sparse (0.1–10% of the footprint) that it almost never samples an address twice, so it can't tell new memory from re-touched memory. Once the true footprint plateaus, the observed footprint keeps rising. Neither the paper's features nor a time or reuse feature fix this. A per-snapshot occupancy (Poisson) estimate gives about −95% error for the same reason.

The evidence that is still available is across bins: how many addresses were seen in exactly one, two, … bins. It would support capture–recapture estimators such as Chao1. The tool does not output it yet.

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
