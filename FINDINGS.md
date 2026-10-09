# Findings: footprint over time

Summary: sampling *addresses* instead of references (`-mode spatial`) estimates the live footprint over time within about 1–10% at 1-in-25 to 1-in-250 addresses on PolyBench, and within 0.2–6% on miniVite with 1, 4 or 16 threads. It needs no training and runs at about the cost of instrumentation alone (miniVite 8192: ~30 s vs ~180 s for a full trace). Everything after "Data" documents the reference-sampling route that led there.

Goal: reconstruct a program's live memory footprint over time from a sparse Pin trace. The data is the timelines written with `-snapshot`/`-track_frees` (see README).

On `main`, the paper's α model applied per snapshot gives 25–55% MAPE. This branch adds per-address sample counts to the Pin tool and tries estimators of the memory that was never sampled.

## Unseen workloads from source code alone (branch `static-generalization`)

Summary: across the 27 PolyBench kernels we tested, an unseen workload's footprint and α can be predicted from its source code alone, more accurately than with its own trained model. Each kernel was held out in turn and predicted from its source and the other 26. All numbers here use the bytes-touched footprint (`-footprint bytes`); PolyBench and miniVite were retraced for it (see "Footprint definition" below).
- **Footprint.** Static analysis plus a runtime baseline fitted on the other kernels predicts the footprint within a median of 0.09% when the largest input is held out (EXTRA), and 0.27% when the middle input is held out (INTER).
- **α.** The α this implies is within a median of 1.2% (EXTRA) and 5.5% (INTER). The kernel's own α model, trained on its own smaller inputs, gets 18.0% and 10.9%.
- **Similarity.** A memory-behaviour descriptor z, predicted from source, ranks which known model transfers best (median Spearman ρ = 0.83–0.85). This is as well as the measured z does (0.84–0.85), and much better than clang AST node counts (0.27–0.31).
- **Irregular code.** On miniVite, GAP and darknet a static access-idiom check flags the code as outside what this covers. On miniVite even the best borrowed PolyBench model is 46% off, so those workloads should use training-free sampling.

### Footprint definition

The first version of these results used the PolyBench and miniVite tables in `~/memory_estimator/tools/data`, traced by the old tool with its default footprint, the largest access per start address. They were retraced here with the current definition, bytes touched:

```
TRACE_DIR=$PWD/data/polybench-bytes/traces RESULTS_DIR=$PWD/data/polybench-bytes/results \
  scripts/run.sh polybench --mode splitter --footprint bytes --runs 1 --bench <kernel>    # 27 kernels, 7 sizes
scripts/run.sh minivite --mode splitter --footprint bytes --runs 1                       # OpenMPI 5.0.5, 1 thread
```

The static spectra are computed for bytes too (`static spectra --footprint bytes`, the default): an access counts once on every byte it covers. For PolyBench, whose accesses are aligned and do not overlap, the two definitions give the same true footprint to within 0.1% (gemm MEDIUM: 1,264,351 B against 1,264,851 B), so the change mostly shows in the bins. The earlier results are kept in `data/polybench-starts/` and `figures/static-starts/`. Under start addresses, static α was 4.5% / 8.0% (EXTRA / INTER median), against 1.2% / 5.5% now; every conclusion below holds under both.

### Why α is decided by the access-count spectrum

The splitter keeps each reference with probability 1/k, independently, and puts it in a bin. An address referenced c times is therefore in a given bin with probability p = 1 − (1 − 1/k)^c. With s_a the size of address a:
- E[m] = Σ s_a p_a
- Var[m] = Σ s_a² p_a (1 − p_a)
- truth = Σ s_a

So the number of references to each address (the spectrum) determines α_k = truth / E[m] and the bins' spread at every interval k. This holds in expectation, without fitting anything. The per-workload regression learns this function of the spectrum implicitly, for one workload at a time.

The memory-behaviour descriptor, for every interval k, has two parts:
- reuse(k) = log α_k / log k. It is 1 when no address is sampled twice and falls towards 0 under heavy reuse.
- spread(k) = log(SD_k / m_k).

### Method

1. **Spectrum from source** (`analysis/memprint/static/interp.py`). A vectorised abstract interpreter runs the program from `main()` over the clang AST, without its data.
   - It tracks integer variables exactly and runs a counted loop's iterations all at once as numpy arrays.
   - It charges every reference an unoptimised (`-O0`, as traced) build makes: locals and parameters in stack slots, array elements, `.rodata` constants, and the return address and frame pointer.
   - Statements under a data-dependent branch are charged half to each arm and counted as uncertain. Addresses that depend on data are counted as unresolved.
   - Coverage is the share of references that are neither.
   - gemm MEDIUM's arrays come out exact (1,158,400 B). Its reference count is 86% of Pin's; the gap is extra `-O0` stack traffic on addresses that are always sampled.
   - A kernel and config takes 0.1–34 s.
2. **Runtime baseline.** What the loader, libc and malloc touch is not in the program text. It is modelled as a shared spectrum: non-negative bytes and addresses at counts 2^0 … 2^26, fitted by NNLS to the known workloads' bin footprints.
   - Fitted on all 27 kernels: 104 KB referenced once, plus 2.9 KB referenced 32–512 times (107 KB in all).
   - In the evaluation it is refitted without the held-out kernel every time.
3. **Idioms for code the interpreter cannot run** (`static/idioms.py`).
   - **References** are classed as affine, indirect (`A[B[i]]`, hash-map lookups), pointer, or other.
   - **Loops** are classed as counted, data-bounded, or while. A loop is data-bounded when its bound is a load that depends on an enclosing loop variable (`row[v+1]`, or an `edge_range(v, e0, e1)` out-parameter).
   - Each reference is weighted by 10^(loop depth).
   - **Gate:** fall back to training-free sampling if interpreter coverage is below 0.5, if the affine share is below 0.95, or if more than 5% of loops are data-bounded or while loops.
4. **Leave one workload out** (`analysis/memprint/lowo.py`). Each held-out kernel C is predicted at its EXTRA (largest) or INTER (middle) config, for every bin at k = 100 … 100000. Only C's source is used; its traces are used only for scoring.

### Accuracy (MAPE of α on the held-out kernel, 27 kernels)

| Method | EXTRA median | INTER median | EXTRA mean | INTER mean |
|---|---|---|---|---|
| static footprint (source + baseline, no sampling) | 0.09 | 0.27 | 0.12 | 0.49 |
| static α (spectrum moments) | 1.19 | 5.50 | 2.99 | 6.97 |
| pooled regression + log static α as a feature | 3.86 | 6.97 | 7.42 | 7.95 |
| nearest known model by z predicted from source (ẑ) | 12.24 | 11.11 | 16.97 | 15.09 |
| RBF mixture of known models by ẑ | 14.06 | 12.29 | 19.28 | 15.46 |
| nearest by measured z (upper bound for ẑ) | 13.84 | 11.19 | 19.45 | 16.01 |
| mixture by measured z | 14.05 | 11.53 | 18.57 | 15.61 |
| nearest by AST node counts | 26.01 | 17.44 | 144.61 | 34.83 |
| uniform mixture (no similarity) | 92.03 | 43.99 | 154.16 | 79.80 |
| C's own model (needs C's traces) | 18.04 | 10.85 | 18.41 | 11.22 |
| best borrowed model, chosen after the fact | 8.89 | 6.05 | 11.72 | 9.20 |

![errors](figures/static/lowo_errors.pdf)

- **Predicting α beats borrowing a model.** Even the best borrowed model, picked after seeing the error, is worse (8.9%) than predicting α from C's spectrum (1.2%).
- **Similarity does help when borrowing.** Picking a model by ẑ cuts the error of a uniform mixture by a factor of 4–8 (11–14% against 44–92%).
- **ẑ is as good as measuring z.** It matches measured z to a median reuse RMSE of 0.0014, at most 0.032 ([curves](figures/static/lowo_reuse_curves.pdf)), and it predicts transfer error as well as the measured z does ([ρ](figures/static/lowo_similarity_validity.pdf)):

| Distance used | ρ EXTRA (median) | ρ INTER (median) |
|---|---|---|
| ẑ | 0.826 | 0.849 |
| measured z | 0.843 | 0.854 |
| AST node counts | 0.268 | 0.307 |

- **Worst static-α cases.** These are floyd-warshall (16.0% / 19.5%) and nussinov (20.1% / 22.5%), the two kernels with data-dependent min/max ternaries. Their interpreter coverage is 0.77; their footprint is still within 0.7% because both arms touch the same arrays. Every other kernel is within 5.5% (EXTRA) and 10.7% (INTER).
- **Worst static footprints** are correlation, gramschmidt and cholesky at their INTER size (1.7–2.6%); everything else is within 0.6%.

### Irregular workloads: the gate

| Program | Affine share | Data-bounded or while loops | Gate |
|---|---|---|---|
| 30 PolyBench kernels | ≥ 0.986 | 0 | in |
| miniVite | 0.79 | 0.53 | out (interpreter coverage 0.06) |
| GAP bfs / pr / tc | 0.72 / 0.68 / 0.20 | 0.53 / 0.63 / 0.90 | out |
| darknet | 0.79 | 0.22 | out |

On miniVite (largest config, 16384), the gate's call is right:
- Its own model gets 11.4%.
- The nearest PolyBench model by measured z (3mm) gets 64.5%. The best PolyBench model chosen after the fact gets 46.3%, and the median PolyBench model 80.0%.
- Its measured z is 4.30 from the nearest PolyBench kernel. PolyBench kernels' own nearest-neighbour distances have a median of 0.79 and a maximum of 1.84.

### Limitations

- **One runtime.** The baseline is shared because every PolyBench kernel uses the same harness, libc and compiler. A workload with another runtime (C++, MPI, OpenMP) needs its own baseline, or the gate.
- **Affine code only.** Irregular workloads get a correct "don't trust this" answer, not a prediction. Predicting their z would need traced irregular workloads to learn from (the plan's synthetic kernels), which this branch does not have.
- **No Polly or islpy.** Neither is installed here. The interpreter enumerates iteration points exactly instead of counting them symbolically, so very large inputs cost time in proportion to their reference count. MEDIUM needed at most 34 s.
- **Not the `-O0` stack traffic of gcc exactly.** The interpreter misses about 14% of the references, all on stack addresses that are always sampled. This does not change α.

### Reproduce

```
pip install -r analysis/requirements.txt    # now includes libclang
export PYTHONPATH=$PWD/analysis
R=data/polybench-bytes          # traces from the retrace above
python -m memprint --root $R preprocess                              # $R/data/<wl>_allData.csv
python -m memprint --root $R static spectra --polybench workloads/src/polybench --jobs 12
python -m memprint --root $R static idioms --polybench workloads/src/polybench \
    --program miniVite=<dir>:<dir>/main.cpp --include <mpi include>  # $R/data/static_idioms.csv
python -m memprint --root $R static lowo --ast ~/memory_estimator/tools/ast_features.csv --transfer miniVite
                                                                     # $R/data/lowo_*.csv, $R/figures/static/
```

## GAP pilot: irregular code from a skeleton and the input's distribution

Summary: for GAP `pr` and `bfs` on uniform random graphs (`-u 10` … `-u 18`, average degree 16), α is predicted within 2.8–8.8% without tracing the kernel. Its own MemPrint model, trained on its other scales, gets 7.3–9.0%; borrowed PolyBench models (bytes-touched traces) get 33–96%. The footprint is predicted within 0.0–1.8%.

**Not a blind test yet.** The skeleton's two largest corrections were found by comparing against these same traces:
- generated graphs are always symmetrized;
- the random-number state uses 8-byte words.

A held-out test, for example Kronecker graphs (`-g`) with no further changes, is still to do.

### Method

- **Workload.** `workloads/gapbs/workload.sh` builds GAP at b5e3e19 single-threaded with its default `-O3`.
  - Benchmarks are `gap_pr` (`-i 20 -t 0`, so exactly 20 iterations) and `gap_bfs`, one trial each.
  - Traced with `scripts/run.sh gapbs --mode splitter --footprint bytes`. The new `--footprint` option selects the bytes-touched footprint.
  - Bytes touched is used rather than start addresses. Under start addresses, GAP's 32-byte vector copies overlapping its 4-byte reads made the footprint 1.7× the bytes touched, which would require modelling instruction widths.
- **Skeleton** (`analysis/memprint/static/skeleton.py`, `static/gap.py`). A numpy replay of `builder.h`, `pr.cc` and `bfs.cc`:
  - The graph is sampled from the generator's distribution, not GAP's actual edges.
  - Each array element is charged its reads and writes. Counts are kept per 4-byte unit, so an access counts once per unit it covers, whatever its width.
  - BFS runs on the sampled graph: top-down claims, bottom-up scans to the first frontier neighbour, and the direction switches.
  - std::sort is costed per neighbour list. A microbenchmark of sort, unique and remove on the same list sizes under Pin measured 68 B of references per entry; the model gives 62.
- **Allocator** (part of the skeleton). glibc-like: a best-fit heap, mappings above a dynamic mmap threshold, and munmap holes not reused. Under Pin, the tool's own mappings take those holes; GAP bfs at scale 18 shows no reuse of its 33.5 MB edge list's addresses.
- **Runtime baseline.** libstdc++, the loader and other runtime memory (about 0.47 MB). It is fitted on the other kernel's traces, so nothing from the predicted kernel is used.

### What the traces showed

| Finding | Effect on the skeleton |
|---|---|
| `command_line.h` sets `symmetrize_ = true` for any generated graph | One CSR of 2M entries shared as in- and out-graph, not two directed CSRs. PR then matches Pin's references per iteration (1.16M vs 1.23M at scale 14), and the footprint error falls from 3.6% to below 0.1% at large scales |
| libstdc++'s `std::mt19937` keeps 624 `uint_fast32_t` state words, 8 bytes each | The state is referenced about 4 × draws / 624 times per word. That is neither rare nor always sampled, so it matters at large k: generation went from 470 to 712 B at k = 10^5, against Pin's 744 B |
| The runtime alone (scale 2) adds 56 B at k = 10^5 | Small at large scales |

After these, the builder phases agree with Pin, phase by phase, to within 4–10% of bin footprint at k = 10^5. Pin's figures come from runs that exit after each phase. The largest remaining miss is degree counting (254 vs 372 B), most likely how Pin counts `lock xadd`.

### Accuracy (MAPE %, EXTRA = scale 18 held out, INTER = scale 14)

| Kernel | split | skeleton footprint | skeleton α | own model | PolyBench nearest by ẑ | PolyBench mean | best PolyBench (oracle) |
|---|---|---|---|---|---|---|---|
| bfs | EXTRA | 0.00 | 5.29 | 7.68 | 93.14 | 93.02 | 54.53 |
| bfs | INTER | 1.65 | 8.78 | 7.28 | 32.82 | 86.61 | 32.82 |
| pr | EXTRA | 0.00 | 2.75 | 8.96 | 95.69 | 85.23 | 84.64 |
| pr | INTER | 1.84 | 4.36 | 8.35 | 80.07 | 78.25 | 60.01 |

α error per scale (all bins at k = 100 … 100000) is 2.8–5.3% for pr and 5.3–8.8% for bfs. The footprint error is at most 0.42% except at scale 14 (1.7–1.8%), where arrays of (N+1) × 8 = 131 KB sit at glibc's 128 KB mmap threshold.

### What generalises, and what was specific

- **Generic, reusable for other programs:** the allocator model, the per-unit counting, the Mersenne Twister and std::sort costs, and the moment formulas.
- **Written for GAP:** the skeleton itself. It is about 250 lines, written by reading the source.

Automating it means extending the interpreter so that loads from input arrays become random variables with a known distribution. The skeleton is the target that such an extension has to reproduce.

### Reproduce

```
PIN_ROOT=... scripts/setup.sh gapbs
TRACE_DIR=$PWD/data/gap-bytes/traces RESULTS_DIR=$PWD/data/gap-bytes/results \
  scripts/run.sh gapbs --mode splitter --footprint bytes --runs 1 --configs "10 11 12 13 14 15 16 17 18"
python -m memprint --root data/gap-bytes preprocess gap_pr gap_bfs
python -m memprint --root data/gap-bytes static gap --borrow data/polybench-bytes/data   # data/gap-bytes/data/gap_pilot_*.csv
```

## Windowed sampling: watching part of a run (branch `windowed`)

Summary: for long runs, the Pin tool can watch memory accesses for a small fraction of the run, as long as it tracks allocations and page residency the whole time. All results below count the footprint as **bytes touched** (see the next section) and were re-measured on 2026-10-08 against new full traces. At 5% watched the live footprint over time is reconstructed:
- within 0.1% on PolyBench LARGE;
- within 3.3–4.0% on average and +1.3% to +1.8% at the peak on miniVite (32768 and 65536 vertices, 1 and 4 threads);
- within 1.7–8.2% on average and −3.7% to +4.1% at the peak on GAP, darknet, Python, Lua, Perl and SQLite, and +6.7% on GCC's cc1.

A heap-churn test, where new blocks land on pages that freed blocks touched, stays within about +23% (page granularity) instead of +271%. Watching nothing at all (pages only) is almost as accurate, except where memory outside heap blocks matters (miniVite 32768: −6.9%). Cost at 5% watched: 3.5×–3.6× native on 2mm and jacobi-2d LARGE (gemm 4.5×–12×, varying between runs), against 30×–53× for full spatial sampling, and 15×–18× on miniVite, against 34×–37×. At 1% watched, which is as accurate, it is 2.2×–2.7× on PolyBench and 13×–17× on miniVite.

### The footprint is the bytes touched

With `-track_frees` or `-window`, the tool counts the bytes a program touches (`-footprint bytes`): the union of every access's byte range, live until released. The paper's definition (`-footprint starts`, still the default without those knobs, so the paper's outputs are unchanged) adds up the largest access at each start address. Overlapping accesses then count more than once: an 8-byte read at every byte of a 1 MB buffer gives 8.5 MB under it and 1.15 MB of bytes touched. SQLite's peak was 3.4× its bytes touched (263 MB against 76 MB), GAP's 1.4× and darknet's 1.5×.

What changed with the definition:
- **The truth** keeps a bitmap of touched bytes per page, and a release clears exactly the released bytes.
- **The allocator's own accesses do not count.** glibc's `malloc.c` code is found by symbol name (public functions and internals such as `_int_free`), and its accesses count as time only. They are chunk headers and free-list links, which land outside live blocks and were never released: a program that allocates and frees 600,000 small blocks ended with 5.7 MB "live", and Lua with 16.7 MB.
- **A freed block releases its whole glibc chunk** (size field, block and slack), read from the chunk header at `free`'s entry and checked against the requested size. String and copy functions read past a block's requested size, and under bytes those bytes otherwise stayed live: Lua ended with 5.3 MB live and nothing allocated, and now ends with 0.3 MB.
- **Spatial and windowed sampling select aligned 8-byte words** and keep the bytes of each access inside selected words, so selected bytes × i is unbiased. An access tests every word it covers: inline for accesses of up to 16 bytes, in a loop for wider ones (known when the instruction is instrumented). A first version selected 64-byte chunks, which is 8× fewer sampling units: spatial error on PolyBench MEDIUM rose from 2.3% to 7.4% at 1-in-100 and from 6.7% to 35% at 1-in-1000.
- **Density is 1 by construction,** so the windowed estimate is the fresh blocks' resident bytes, plus the reused blocks' written bytes, plus the windows' selected bytes outside blocks × i.
- Default splitter and sampler outputs are byte-identical to the previous tool (checked with address randomisation off); the tests pass.

### Validation against Valgrind and the operating system

Every workload's true peak (full trace, bytes touched) and the tool's tracked live allocations were compared with native peak RSS (`/usr/bin/time`), Massif (heap, and `--pages-as-heap`) and DHAT (heap at its global maximum, t-gmax). Single thread, same inputs, MB:

| Workload | True peak (bytes touched) | Tracked live allocations | Massif heap | DHAT at t-gmax | Massif heap admin | Massif pages | Peak RSS |
|---|---|---|---|---|---|---|---|
| 2mm LARGE | 37.1 | 37.1 | 37.0 | 37.0 | 0.0 | 43.5 | 38.3 |
| gemm LARGE | 29.1 | 29.0 | 29.0 | 29.0 | 0.0 | 35.4 | 30.1 |
| jacobi-2d LARGE | 27.1 | 27.1 | 27.0 | 27.0 | 0.0 | 33.5 | 28.3 |
| miniVite 65536 | 28.2 | 52.0 | 43.2 | 43.2 | 0.4 | 206.2 | 47.5 |
| miniVite 32768 | 15.4 | 30.7 | 23.5 | 23.5 | 0.4 | 206.2 | 35.6 |
| GAP bfs | 72.8 | 72.5 | 72.4 | 72.4 | 0.0 | 95.5 | 76.1 |
| GAP pr | 72.5 | 72.5 | 72.4 | 72.4 | 0.0 | 95.5 | 76.2 |
| GAP cc | 72.8 | 72.5 | 72.4 | 72.4 | 0.0 | 95.5 | 76.2 |
| GAP sssp | 133.8 | 135.5 | 135.4 | 135.4 | 0.0 | 158.5 | 139.2 |
| darknet AlexNet | 269.9 | 531.5 | 531.5 | 532.2 | 0.2 | 548.5 | 285.7 |
| python | 95.7 | 98.5 | 94.7 | 95.1 | 26.4 | 383.3 | 114.2 |
| lua | 55.5 | 52.4 | 52.4 | 52.4 | 5.8 | 86.4 | 65.7 |
| perl | 54.9 | 53.6 | 54.7 | 54.7 | 3.9 | 304.2 | 64.6 |
| sqlite | 76.2 | 75.6 | 78.4 | 78.4 | 0.4 | 102.6 | 82.5 |
| cc1 `-O0` | 105.4 | 113.9 | 16.3 | 16.3 | 0.5 | 384.8 | 134.3 |
| churn, 25% touched | 1.3 | 4.8 | 4.9 | 4.9 | 0.0 | 11.6 | 6.2 |

- **The truth never exceeds the peak RSS** (21–97% of it). It should not: RSS counts whole pages, code and libraries, and memory freed but still resident. Where a program touches what it allocates (PolyBench, GAP), the truth is 95–97% of the RSS.
- **The tool's allocation tracking agrees with Massif and DHAT** to within 0.1 MB on PolyBench, GAP, darknet and Lua, and within 1–3 MB on Perl and SQLite (Massif reports the exact peak, the tool the largest snapshot). It is higher where programs map memory themselves, which Massif's heap mode does not count: cc1 (+98 MB of GCC's collector pages), miniVite (+7 to +9 MB, MPI buffers) and Python (+3.8 MB, pymalloc arenas).
- **Allocated is not touched.** darknet allocates 532 MB and touches 270 MB; miniVite touches about half of its heap; the churn test a quarter. Massif and DHAT measure allocation, so they over-state those programs' footprint by 1.5–4×, while the tool's truth stays below the RSS.
- DHAT's per-offset access counts (kept only for blocks of up to 1 KB) cover too little of these heaps to check bytes touched directly.

### MemPrint against Massif and DHAT: figures

`paper/valgrind_figures.py --scratch <dir>` draws three figures (`paper/figures/*_vs_valgrind.pdf`). Massif's heap peak and DHAT's heap at its global maximum were identical in all 189 PolyBench runs and within 0.7 MB (0.4%) in the other 16, so the figures draw them side by side.

**Static (source-only) footprints** (`static_vs_valgrind.pdf`): the static predictor of the `static-generalization` work (source spectrum plus a runtime baseline fitted on the other kernels, leave one workload out) against the splitter's whole-run footprint (paper's definition, `data/*_allData.csv`), for all 27 kernels at all 7 sizes, with Massif, DHAT and peak RSS run on the same binaries:
- **Static** is within a median of 0.09% at MEDIUM and 0.30% over all sizes (0.994–1.002× at MEDIUM).
- **Massif and DHAT** see only the arrays: 0.89–0.93× of the footprint at MEDIUM (a median 107 KB short, the loader, libc and stack that the static runtime baseline models), and 0.07–0.20× at MINI, where that runtime memory dominates.
- **They overshoot where a kernel allocates more than it touches:** trisolv (1.70×) and trmm (1.08×) allocate full N×N matrices and use one triangle.
- **Peak RSS** is 1.3–2.9× at MEDIUM (a median 1.2 MB above the footprint over all sizes): code, libraries and whole pages.
- Depends on the static code and spectra in the working tree (`analysis/memprint/lowo.py`, `static/`, `data/static/`), which are not committed on this branch; the prediction script is in the scratch space (`cmp/static_pred.py`).

**Timeline peaks** (`timeline_peaks_vs_valgrind.pdf`): the 16 workloads of the validation table, peak / true live peak (bytes touched):

| Workload | MemPrint windowed, 5% | Massif = DHAT | Peak RSS | Massif pages |
|---|---|---|---|---|
| 2mm / gemm / jacobi-2d LARGE | 1.00 | 1.00 | 1.03–1.04 | 1.17–1.23 |
| GAP bfs / pr / cc / sssp | 1.00–1.01 | 1.00–1.01 | 1.04–1.05 | 1.18–1.32 |
| miniVite 65536 / 32768 | 1.01–1.02 | 1.53 | 1.68–2.31 | 7.3–13.4 |
| darknet | 1.04 | 1.97 | 1.06 | 2.03 |
| Python / Lua / Perl / SQLite | 0.96–1.04 | 0.94–1.03 | 1.08–1.19 | 1.35–5.54 |
| GCC cc1 | 1.06 | 0.15 | 1.27 | 3.65 |
| heap churn | 1.23 | 3.84 | 4.83 | 9.07 |

- The windowed estimate is within −4% to +6% of the true peak everywhere except the churn test (+23%, page granularity).
- Heap profilers agree with the truth only where a program touches what it allocates. They are 1.5–3.8× too high on miniVite, darknet and the churn test, and 0.15× on GCC, whose collector maps its own pages.
- **Over time** (`timeline_curves_vs_valgrind.pdf`; Massif's time axis is instructions, the truth's memory references, both as fractions of the run): Massif's heap is a step above the truth as soon as an array is allocated (2mm's second array is touched halfway through), stays at the allocation on darknet and the churn test, and follows the truth on Lua and SQLite. DHAT's maximum lands at the same moment as Massif's: 25% into darknet's run, 88% into miniVite's, 49% into Lua's.

### Why

Spatial sampling still checks every memory access, even when it records only 1 in 100 words. On 2mm LARGE that alone takes 284 s against 8.0 s native, and Pin itself costs almost nothing (8.2 s). For long real workloads that overhead is the obstacle, not the sampling rate.

### Method

`-mode spatial -window W -period P` watches accesses for W of every P memory references.

**Between windows,** only two things run:
- a reference counter, one inlined add per basic block, which keeps time;
- the allocator hooks.

**Live allocations are tracked all the time.** These are heap blocks (the malloc family) and anonymous mmaps. Frees remove them.

**At every snapshot (`-snapshot N`), the estimate is:**

    Σ over live blocks (touched bytes)
    + windows' selected bytes outside blocks × i

- **Touched bytes:** the bytes of a block on resident pages, read from `/proc/self/pagemap`. A page becomes resident when it is first read or written, so residency records first touches whether or not a window is open.
- **Fresh blocks** got pages mapped for them: an anonymous mmap, or a malloc that made a new mmap. Their pages start untouched, so residency is exact to the page.
- **Reused blocks** sit on heap pages that may have been used before, so their residency can include pages that earlier blocks touched. They are measured instead by the pages *written* since they were allocated, from the kernel's soft-dirty bits (pagemap bit 55, cleared by writing `4` to `/proc/self/clear_refs`). Heap memory is written before it is read, so written pages are its touched pages.
  - Clearing write-protects every page, so the next write to each page faults once. Bits are therefore cleared at three points rather than at every snapshot.
    - At the allocator's entry, for allocations of 16 KB or more, after recording every reused block's written pages. Writes to the new block's pages by earlier blocks then do not count, and calloc's zeroing and realloc's copy, which happen inside the call, do.
    - Around `brk`, `mmap` and `mremap`: pagemap reports a whole region as soft-dirty when the kernel creates or grows it, whatever was written. The bits are recorded at the system call's entry and cleared at its exit.
    - Every `-dirty_every` references (default: every 10 snapshots).
  - Blocks whose pages are all marked written are skipped when recording, and realloc in place keeps a block's marks.
- **Huge pages:** the tool disables transparent huge pages for the traced process (`prctl(PR_SET_THP_DISABLE)`). Without that, a single touch makes a whole 2 MB region resident.
- **Memory outside blocks** (stack, globals, MPI shared memory) is seen only in windows.
- **Allocation windows** (`-window_alloc N`, default 1 MB; 0 turns them off): an allocation of at least N bytes opens a window at once, at most one per period on top of the periodic windows, so at most twice the periodic fraction is watched. A periodic window that starts inside such a window extends it.
- **With `-footprint starts`** the estimate multiplies each block's touched bytes by a density (footprint ÷ covered bytes, measured in windows on 64-byte chunks). That correction, and its failure modes, are described in "Under the paper's definition" below.

**Output:** each snapshot is a row of `<prefix>_windowed.csv` with every component. `analysis -m memprint timeline windowed` scores the runs against the splitter's truth.

### Accuracy

Ground truth comes from full traces (splitter with `-track_frees`) of the same input and thread count. Each windowed run is scored on its whole timeline, with time as a fraction of the run. Settings: 1-in-100 words, about 20 windows per run, periodic windows only unless stated, nothing tuned per workload. Each cell is mean error / peak error, %.

**PolyBench LARGE** (2mm, gemm, jacobi-2d; live peaks of 27–37 MB, 28–49 billion references):
- The windowed estimate is within 0.0–0.1% mean and 0.1% at the peak at every watched fraction, including none. Every array is fresh, so residency alone gives the exact curve.
- Baselines:
  - live allocated bytes: 0.4–4.5% mean error;
  - the windows' selected bytes alone at 5% watched: 0.1–10.9%;
  - the same at 1% watched: 10.9–46.7%.

**miniVite** (Louvain on random geometric graphs; its footprint grows in reused heap blocks):

| Input, threads | True peak | Nothing watched | 1% | 5% | 20% | 5%, 1-in-25 | Watching all the time |
|---|---|---|---|---|---|---|---|
| 65536, 1 thread | 28.2 MB | 10.1 / −3.8 | 2.8 / +1.1 | 3.9 / +1.5 | 3.7 / +1.4 | 3.6 / +1.3 | 3.6 / +1.4 |
| 65536, 4 threads | 28.2 MB | 9.8 / −3.7 | 2.4 / +0.9 | 3.6 / +1.3 | 4.2 / +1.6 | 3.7 / +1.4 | 3.8 / +1.4 |
| 32768, 1 thread | 15.4 MB | 14.7 / −6.9 | 1.0 / +0.5 | 3.3 / +1.5 | 5.3 / +2.5 | 3.8 / +1.8 | 4.9 / +2.3 |
| 32768, 4 threads | 15.4 MB | 14.6 / −6.9 | 1.8 / +0.8 | 4.0 / +1.8 | 5.3 / +2.5 | 3.7 / +1.7 | 5.7 / +2.7 |

Baselines at 5% watched (65536, 1 thread):
- residency without written pages: 4.0 / +1.5, the same as the estimate (miniVite's reused blocks are written all over);
- the windows' selected bytes alone: 50.5 / −36.7;
- live allocated bytes: 107.0 / +84.6. miniVite allocates far more than it touches.

- **Residency is necessary.** Windows alone see only accesses that happen inside a window. They miss memory that is touched once and then left alone.
- **Watching matters only for memory outside blocks.** With nothing watched, the peak is 4–7% low, because the stack, globals and MPI memory are not seen. 1% watched is enough, and more watching does not help: the estimate creeps up by 1–2 points between 1% and 20%, as whole pages and selected words outside blocks accumulate.
- **Threads.** 1 and 4 threads agree within a point on both inputs.
- Under the paper's definition the same runs were 1.7–17.1% / −0.4% to −15.2% at 1–20% watched, and 26–32% / −25% to −28% with nothing watched, because density had to be measured in windows.

**Heap churn** (`tests/pintool/churn.c`): 64 live heap blocks of 16–112 KB, each freed and replaced in turn, of which only the first quarter (or half) is written. 5% watched:

| Fraction of each block touched | True peak | Live allocated bytes | Residency | Windowed (written pages) |
|---|---|---|---|---|
| 25% | 1.3 MB | 261.6 / +273.5 | 249.4 / +270.6 | 24.1 / +22.8 |
| 50% | 2.5 MB | 92.1 / +93.4 | 91.4 / +89.9 | 12.9 / +12.2 |
| 100% | 4.9 MB | 2.2 / −1.6 | 0.3 / +0.1 | 0.2 / +0.1 |

- Residency fails because freed blocks' pages stay resident and are handed to later blocks.
- The windowed estimate's remaining error is page granularity: each block's touched prefix ends inside a 4 KB page, which adds up to 4 KB per block (64 blocks, about 0.25 MB).
- It is the same at every watched fraction (22.7–25.5% for 25% touched), because written pages, not windows, carry the estimate.

### Windows opened by large allocations

5% and 1% watched, 1-in-100 words, about 20 periodic windows; mean error / peak error, %:

| Run | Periodic only, 1% | With allocation windows, 1% | Periodic only, 5% | With allocation windows, 5% | Allocation windows opened |
|---|---|---|---|---|---|
| miniVite 65536, 1 thread | 2.8 / +1.1 | 3.3 / +1.3 | 3.9 / +1.5 | 3.6 / +1.4 | 7 / 6 |
| miniVite 65536, 4 threads | 2.4 / +0.9 | 2.4 / +0.9 | 3.6 / +1.3 | 3.4 / +1.3 | 7 / 6 |
| miniVite 32768, 1 thread | 1.0 / +0.5 | 1.6 / +0.7 | 3.3 / +1.5 | 4.2 / +2.0 | 5 / 5 |
| miniVite 32768, 4 threads | 1.8 / +0.8 | 1.6 / +0.7 | 4.0 / +1.8 | 5.0 / +2.3 | 5 / 5 |

- With bytes touched, allocation windows change nothing that matters: they were there to measure the density of large blocks written late in a run, and density is now 1. PolyBench (arrays allocated inside the first window) and the churn test (blocks under 1 MB) open no extra windows.
- Under the paper's definition they mattered, and the diagnosis is kept below ("Under the paper's definition").

### GAP and darknet as heap-reuse workloads

MemGaze used GAP (graph kernels) and darknet (neural networks). Copies are in `~/memory_estimator/workloads`. Composition at the true peak (5% watched, allocation windows; MB):

| Run | Fresh blocks (resident) | Reused blocks (resident / written) | Residency − written, max over the run |
|---|---|---|---|
| GAP bfs `-g 18 -n 8` | 71.4 (71.4) | 1.1 (1.1 / 1.1) | 1.1 |
| GAP pr `-g 18 -n 4` | 71.4 (71.4) | 1.1 (1.1 / 1.1) | 1.0 |
| GAP cc `-g 18 -n 8` | 71.4 (71.4) | 1.1 (1.1 / 1.1) | 1.0 |
| GAP sssp `-g 18 -n 4` | 130.2 (128.1) | 5.3 (5.3 / 5.3) | 1.0 |
| darknet AlexNet, 4 images, random weights | 518.7 (268.6) | 12.1 (12.1 / 12.1) | 1.1 |

- Neither is a heavy heap-reuse workload in the sense that breaks residency. Their large buffers are above glibc's mmap threshold (128 KB, raised dynamically up to 32 MB after frees), so they get new pages, and the reused blocks they do have are fully rewritten.
- Forcing large blocks onto the heap with `MALLOC_MMAP_THRESHOLD_=33554432` (glibc ignores values above 32 MB) moves more memory into reused blocks (GAP 5–7 MB, darknet 86 MB) but residency still exceeds written pages by at most 1 MB. (Measured with the earlier tool; residency and written pages do not depend on the footprint definition.)
- They are still useful real workloads: darknet allocates 532 MB but touches 270 MB, so allocation-based estimates are 2× off, and GAP has large, irregularly accessed graphs.

**Accuracy on GAP and darknet** (single thread unless stated; mean error / peak error, %):

| Run | True peak | Allocated bytes | Nothing watched | 1%, allocation windows | 5% | 5%, allocation windows | 20%, allocation windows | Watching all the time |
|---|---|---|---|---|---|---|---|---|
| GAP bfs | 72.8 MB | 201.9 / −0.3 | 5.0 / −0.3 | 3.1 / −0.0 | 2.2 / +0.2 | 2.1 / +0.2 | 2.5 / +0.2 | 2.3 / +0.2 |
| GAP pr | 72.5 MB | 143.8 / −0.0 | 3.8 / −0.0 | 1.8 / +0.4 | 1.7 / +0.5 | 1.7 / +0.5 | 1.6 / +0.5 | 1.5 / +0.5 |
| GAP cc | 72.8 MB | 198.7 / −0.3 | 4.9 / −0.3 | 3.0 / −0.0 | 2.2 / +0.2 | 1.9 / +0.1 | 2.3 / +0.2 | 2.0 / +0.2 |
| GAP sssp | 133.8 MB | 183.3 / +1.3 | 3.6 / −0.2 | 2.4 / +0.1 | 2.4 / +0.1 | 2.3 / +0.1 | 2.2 / +0.0 | 2.3 / +0.0 |
| GAP bfs, 4 threads | 72.8 MB | 419.8 / +34.2 | 4.7 / −0.3 | 3.0 / −0.0 | 2.7 / +0.2 | 2.5 / +0.2 | 2.5 / +0.2 | 2.4 / +0.2 |
| GAP pr, 4 threads | 72.6 MB | 337.6 / +34.6 | 8.3 / −0.0 | 11.7 / +0.4 | 5.6 / +0.6 | 6.4 / +0.5 | 4.5 / +0.6 | 4.1 / +0.5 |
| darknet AlexNet | 269.9 MB | 153.2 / +96.9 | 3.1 / +4.0 | 3.2 / +4.6 | 3.2 / +4.1 | 3.2 / +4.1 | 3.2 / +4.1 | 3.2 / +4.1 |

- Every run is within +4.6% of its peak at every watched fraction, including none.
- GAP's allocated bytes match the truth at the peak but not over the run (144–420% mean error): GAP allocates its graph early and fills it later.
- darknet's +4% is fresh pages touched only in part.
- Under the paper's definition these peaks were 9–28% low at 5% watched, because the windows measured too little of the arrays' density (GAP bfs: 1.08 at 5% against 1.36 over the whole run).

**darknet exposed a bug.** The tool recorded 53 KB of darknet's 513 MB. glibc's `calloc`, the first time the allocator is used, calls `malloc` through an initialization function and leaves with a jump to `memset`, so its exit is never seen. The tool then treated every later allocation made from deeper in the stack as nested inside it and recorded none. A call now counts as nested only if it comes from inside an instrumented allocation function and runs deeper in the stack, and a block returned by a call nested in a malloc-like call is recorded at once. `tests/pintool/first_calloc.c` checks this; default outputs are unchanged. miniVite, GAP and PolyBench were not affected, because their first allocation is a `malloc`.

### Interpreters, a database and a compiler

Six programs that churn their heap, run under Pin with the system's binaries; native time and memory references in brackets:
- `python3` 3.6 serialising, parsing and sorting records in rounds of 10k–90k (3.5 s, 10.5 G);
- `lua` 5.3 growing tables and concatenating strings in rounds (2.5 s, 4.7 G);
- `perl` 5.26 doing the same with hashes and strings (1.4 s, 2.4 G);
- `sqlite3` 3.26, an in-memory database of 300k rows with indexes, deletes, a join and `VACUUM` (0.9 s, 2.5 G);
- GCC 8's `cc1` compiling the SQLite amalgamation at `-O0` (1.6 s, 2.8 G) and `-O1` (4.6 s, 7.3 G).

The scripts are in the session's scratch space, not in the repository.

**They are heap-reuse workloads.** Lua, Perl and SQLite keep 84–100% of their live memory in reused heap blocks (44, 46 and 73 MB at the peak, in 24k–650k blocks). Python and cc1 keep most of theirs in fresh blocks: pymalloc's 256 KB arenas and GCC's garbage-collected pages are mapped with `mmap`.

**Residency still does not break.** The reused blocks' residency exceeds their written pages by at most 1.2 MB at any point of any run. The blocks are reused, but each new block is written all over.

**Accuracy** (mean error / peak error, %; allocation windows on unless stated):

| Program | True peak | Allocated bytes (5%) | Nothing watched | 1% | 5% | 5%, periodic only | 20% | Watching all the time |
|---|---|---|---|---|---|---|---|---|
| python | 95.7 MB | 3.4 / +3.0 | 3.4 / +2.8 | 4.1 / +3.8 | 3.5 / +3.8 | 4.0 / +3.8 | 4.5 / +4.0 | 4.9 / +4.0 |
| lua | 55.5 MB | 5.8 / −5.6 | 5.9 / −5.4 | 6.6 / −4.8 | 6.2 / −3.7 | 6.7 / −4.0 | 8.1 / −0.7 | 11.8 / +5.2 |
| perl | 54.9 MB | 1.7 / −2.3 | 1.9 / −3.5 | 1.9 / −2.9 | 1.9 / −2.7 | 1.7 / −2.8 | 1.8 / −1.9 | 1.7 / −0.6 |
| sqlite | 76.2 MB | 16.9 / −2.9 | 6.4 / −0.7 | 8.5 / −2.4 | 8.2 / −2.5 | 8.2 / −2.8 | 8.4 / −2.4 | 8.4 / −0.8 |
| cc1 `-O0` | 105.4 MB | 11.5 / +8.3 | 3.6 / +3.5 | 8.1 / +6.4 | 8.5 / +6.7 | 8.5 / +6.3 | 8.7 / +6.7 | 8.9 / +6.6 |

- **Every peak is within −5.4% to +6.7%,** at every watched fraction. Under the paper's definition, before the fixes below, the same programs were −10% to −66% at the peak at 5% watched.
- **These programs touch what they allocate,** so live allocated bytes are about as good as the estimate (except SQLite's mean error and cc1, whose collector reserves pages it has not used yet).
- **Watching adds a little on cc1 and Lua.** cc1 is +3.5% at the peak with nothing watched and +6.4% to +6.7% with windows; Lua drifts from −5% to +5% as more is watched. The windows' selected bytes outside blocks then exceed what is live there; not investigated further.
- **cc1's `-O1` windowed runs were not made:** at `-O0` they take 11–16 minutes (see Cost).

What went wrong under the paper's definition, before 2026-10-08, and why it no longer applies:
1. **Density accumulates over a run.** Each new access pattern adds start addresses to memory already covered, so a window measures less density than the whole run has (Lua: 1.11 at 5% against 1.36; GAP: 1.08 against 1.36). Bytes touched has no density.
2. **The ground truth counted allocator metadata as live memory** (15% of Lua's peak, 7% of Perl's, 16% of SQLite's). Fixed: the allocator's own accesses no longer count, and a free releases the whole chunk.
3. **SQLite's footprint was not its memory**: 263 MB against 76 MB of bytes touched and an 80 MB process. Bytes touched removes the overlap.

### Under the paper's definition: the allocation-window diagnosis

Measured before the switch to bytes touched, kept because it explains what density did. Single-thread miniVite 65536 at 5% watched gave −2.6% at the peak with periodic windows and −4.4% with allocation windows; watching all the time gave −1.5%. A debug build wrote every block of 256 KB or more at each snapshot. At the true peak, per block:

| 5% watched | Fresh blocks: estimate − full watch | Reused blocks: estimate − full watch | Peak error |
|---|---|---|---|
| Periodic only | −3.75 MB | +3.09 MB | −2.6% |
| With allocation windows | +0.03 MB | −0.95 MB | −4.4% |

- **Periodic only:** a 25 MB fresh block allocated at 88% of the run was not inside any window and counted at density 1.00 against its true 1.28 (−3.75 MB); a 9 MB reused block allocated at 94% took the pooled density of reused blocks, 1.44, against its true 1.00 (+4 MB). The errors cancelled.
- **Allocation windows** measured both blocks exactly. The rest was a few 0.3–0.5 MB reused blocks allocated at the start, whose true density (1.9–2.4) comes from overlapping accesses throughout the run.
- **miniVite 32768** had the same structure without the luck: −10.1% periodic, −5.7% with allocation windows, −1.9% watching everything.
- The same run's peak, counted as bytes touched, is now +1.5% (periodic) and +1.4% (allocation windows) at 5% watched.

### Cost

Seconds, quiet machine, one run each (the median where a run was repeated), 1-in-100 words, about 20 windows, periodic windows only:

| Workload | Native | Pin, no tool | Full spatial | Nothing watched | 1% | 5% | 20% |
|---|---|---|---|---|---|---|---|
| 2mm LARGE | 8.0 | 8.2 | 284 | 15.7 | 18.1 | 29.0 | 67.7 |
| gemm LARGE | 5.1 | 5.4 | 271 | 11.5 | 13.9 | 48.1 | 56.6 |
| jacobi-2d LARGE | 11.3 | 11.5 | 344 | 21.6 | 25.0 | 39.8 | 93.3 |
| miniVite 65536, 1 thread | 6.4 | 11.7 | 216 | 67.4 | 81.2 | 96.1 | 121 |
| miniVite 65536, 4 threads | 6.0 | 11.1 | 221 | 84.1 | 101 | 109 | 136 |

- **Full traces** for the truth took 3116–4080 (PolyBench LARGE, miniVite 65536) and 1107–1169 (miniVite 32768) s. Those ran concurrently, so they are not comparable to this table.
- **PolyBench:** with nothing watched, the cost is the per-block reference counter, 1.9×–2.3× native. Each further 1% watched adds about 0.8% of the cost of full spatial sampling over that floor.
- **Word selection** tests one or two words per access (two or three for accesses of 9–16 bytes), against one start address with `-footprint starts`; full spatial sampling on 2mm LARGE takes 284 s, against 211 s with the paper's definition (earlier measurement). A first version whose test Pin could not inline took 330 s.
- **Outliers:** one run of full spatial sampling on miniVite 65536 with 1 thread took 723 s and a repeat 216 s, which the table shows; one of three identical spatial runs of 2mm MEDIUM took 23.5 s against 3.5 s (medians are reported). The earlier tool had the same kind of outlier (jacobi-2d at 5% watched: 173 s, then 31 s). Windowed runs vary too: gemm LARGE at 5% watched took 48, 61 and 23 s in three runs (median shown), whereas 2mm (four runs, 29–35 s, median shown) and jacobi-2d (four runs, 39.6–39.8 s) were steady.
- **Soft-dirty tracking** cost about 14% on miniVite at 5% watched and 4–14% on PolyBench in an earlier measurement. With 1 thread miniVite clears the bits 8509 times; recording only blocks not yet fully written brought that run from 138 s down to 86 s.
- **miniVite** has a higher floor. It makes 8 million malloc/free calls at 65536, each hooked and recorded under a lock. On miniVite 16384, turning the allocator hooks off saved about 7 s of 22 (earlier tool).
- **Allocation-heavy programs** (concurrent runs, so rough): Python, Lua, Perl and SQLite took 24–91 s with nothing watched, 52–216 s at 5% and 95–364 s watching everything, against 450–1684 s for their full traces.
- **cc1 is pathological with windows:** 101–122 s with nothing watched and 182–185 s watching everything, but 658–963 s with windows (1–20% watched), against 689 s for the full trace. Opening or closing a window discards Pin's translated code (`PIN_RemoveInstrumentation`), and cc1 has a lot of code to translate again; with 20 windows that happens 40 times. An earlier note blamed soft-dirty clears around `mmap`; these timings rule that out.

### Implementation notes

- **Switching windows.** `PIN_RemoveInstrumentation` doesn't change code that is running: a loop keeps jumping back into its old translation. In 2mm, a window that had "closed" kept sampling for the rest of the loop nest.
  - Each thread therefore checks the window state every few thousand references. If it changed, the thread restarts at its current instruction (`PIN_ExecuteAt`) in freshly instrumented code.
  - Accesses recorded by stale code after a window closes are dropped.
- **Reference counting.** Counting per basic block matches the splitter's per-access count to within 0.0001% on miniVite (1.78 billion references). A REP string instruction counts once, not once per iteration.
- **Bytes per block** come from the word sample, sorted, as the union of the selected accesses' byte ranges.

### Limitations

- **Written pages miss reads.** A reused block is measured by the pages written since it was allocated, so memory it only reads (rare for heap memory) is missed. Blocks under 16 KB do not clear the bits when allocated, so they can inherit marks from earlier writes since the last clear; they share pages with other blocks anyway. With several threads, writes made by other threads between a `brk`/`mmap` system call's entry and exit are lost.
- **Page granularity:** up to 4 KB per block, which dominates for programs with many small, partly touched blocks (about +23% on the churn test), and for fresh blocks touched only in part (darknet +4%).
- **Not tracked:** `mremap` outside realloc, `madvise`, and stack frames. Memory outside blocks is estimated from windows only.
- **Allocator:** the exclusion of allocator accesses and the chunk-sized releases are specific to glibc's malloc, found by symbol name (libc's symbol table). Other allocators (jemalloc, tcmalloc) are neither tracked as blocks nor excluded.
- **Selection is by absolute address,** so spatial and windowed outputs change between tool builds (the app's mmap addresses move). Splitter and sampler outputs stay byte-identical.
- **Window switching retranslates code**, which is expensive for programs with a lot of code (cc1).

### Reproduce

    scripts/run.sh polybench --bench 2mm --configs LARGE --mode spatial -i 100 -s 20 --runs 1 \
        --snapshot 7000000 --window 85000000 --period 1700000000      # 5% watched, ~20 windows
    scripts/run.sh polybench --bench 2mm --configs LARGE --mode splitter --runs 1 \
        --intervals 1000 --bins 1 --snapshot 7000000 --track-frees     # truth
    python -m memprint timeline windowed 2mm                           # from analysis/

## Headline: spatial (address) sampling

`-mode spatial -i R -s 20` selects 1-in-R aligned 8-byte *words* with a salted hash and records the bytes of every access inside a selected word. A byte is then in the sample with probability 1/R whatever its access pattern, so the footprint estimate is simply the selected bytes × R, with no model and no training. The 20 hash buckets give each snapshot a standard error. (With `-footprint starts` the unit is the start address, as before.)

This removes the core difficulty of all the reference-sampling estimators below. With reference sampling, an address accessed r times is seen with probability 1 − (1 − p)^r, and r is unknown and grows with input size.

Selected accesses are rare, so they are recorded immediately, under a lock, as they happen; frees are applied as they happen too. The order across threads is therefore exact, and time is the true count of references executed.

### PolyBench

Held-out runs (largest and middle size of 2mm, gemm, jacobi-2d, atax; truth from the splitter); mean absolute error, extrapolation / interpolation, %:

| Method | MAPE | Error of peak | Error at peak | Runtime, MEDIUM (2mm / gemm / jacobi-2d) |
|---|---|---|---|---|
| Spatial 1/25 words | 1.6 / 4.0 | 1.5 / 4.0 | 1.5 / 4.0 | 5.7 / 3.6 / 4.2 s |
| Spatial 1/100 | 2.8 / 4.9 | 2.9 / 4.5 | 2.9 / 4.0 | 3.5 / 3.2 / 3.4 s |
| Spatial 1/250 | 2.9 / 4.1 | 2.9 / 3.5 | 3.0 / 3.5 | 3.4 / 3.1 / 3.3 s |
| Spatial 1/1000 | 5.3 / 16.0 | 5.2 / 15.2 | 21.8 / 11.6 | 3.3 / 3.1 / 3.2 s |
| Best reference-sampling hybrid, `-i 25` | 23.5 / 10.0 | 25.0 / 17.5 | 37.3 / 26.3 | 7.1 / 5.9 / 7.6 s |
| Best reference-sampling hybrid, `-i 3` | 4.4 / 4.2 | 3.4 / 7.0 | 11.2 / 7.9 | 28.2 / 20.9 / 35.7 s |
| Splitter (full trace) | — | — | — | 71.4 / 50.5 / 90.6 s |

- **Accuracy follows the binomial prediction,** relative error ≈ 1/√(words / R).
  - The largest sizes (1–1.6 MB, about 130–200k words) stay within about 5% down to 1/1000.
  - The middle sizes (about 0.22 MB, about 28k words) need 1/250 or denser. At 1/1000 only about 28 of their words are selected.
  - The error is largest early in a run, when the footprint is still tiny.
- **The bucket error bar is honest at dense rates.** The truth lies within ±2 standard errors in 99% of snapshots at 1/25 and 1/100 for the largest sizes, but only in 75% at 1/25 for the middle sizes and at 1/250 for the largest (nominal 95%; a 20-bucket standard error is itself noisy).
- Under the paper's definition (start addresses) the same rates gave 0.9 / 3.4, 2.3 / 4.2, 2.3 / 9.9 and 6.7 / 17.5 MAPE.

### miniVite and multithreading

miniVite was rebuilt against spack's OpenMPI 5.0.5, the MPI behind the paper's runs, and run with `OMP_WAIT_POLICY=passive`.
- **Footprint:** a full trace of 1024 vertices gives 6.8–7.0 MB cumulative footprint under the paper's definition, matching the paper's 6.84 MB. With the system OpenMPI it was ~175 MB, mostly MPI start-up memory.
- **References:** 44 million instead of the paper-era 1.76 billion. About 97% of those were OpenMP threads spinning while they waited.

Every spatial run is compared with the full trace of the same size and the same thread count, with no training. The live peak with `-track_frees`, as bytes touched, is 3.8, 4.8 and 6.3 MB at 1024, 4096 and 8192 vertices, the same at every thread count.

| Threads | Spatial 1/25 | 1/100 | 1/250 | 1/1000 |
|---|---|---|---|---|
| 1 | 0.5–1.3% | 0.4–3.8% | 1.5–5.9% | 2.4–5.8% |
| 4 | 0.3–0.7% | 0.7–3.6% | 1.1–2.3% | 4.5–6.8% |
| 16 | 0.5–0.7% | 0.7–2.2% | 0.7–2.1% | 1.0–5.8% |

(MAPE range over the three sizes.)

- **Peaks hold under threads.** Error of peak is ≤ 7.7% in every case and ≤ 3.6% at 1/25 and 1/100.
- **The error bars are not always honest.** The truth lies within ±2 standard errors in 98–100% of snapshots for most runs, but for a few (1 thread at 1/25 to 1/250, 4 and 16 threads at 1/100) in only 1–69%: those runs are off by more than their 20 buckets suggest. Under the paper's definition the coverage was 94–100%.
- **Cost, 8192 vertices:**

  | Threads | Native | Splitter | Spatial (1/25 … 1/1000) |
  |---|---|---|---|
  | 1 | 0.22 s | 187 s | 25–36 s |
  | 4 | 0.17 s | 175 s | 25–39 s |
  | 16 | 0.16 s | 177 s | 25–41 s |

  The per-access lock is not a bottleneck at these rates.
- **Multithreading bugs found and fixed.** `tests/pintool/parallel.c` has 8 threads with 12 MB live at a barrier.
  - Buffered modes kept each thread's accesses in its own trace buffer, so a free in one thread could be applied before another thread's earlier accesses: the full-trace peak came out at 9.8–10.8 MB.
  - With `-snapshot`/`-track_frees` the buffer is now 16 pages, giving 12.08 MB on every run; default outputs are unchanged.
  - A first spatial fix held frees back with vector clocks. It was correct, but on miniVite, which frees constantly while MPI helper threads rarely flush, it took 7474 s. Recording spatial accesses immediately replaced it: 12 runs give 12.03–12.27 MB, with 28–40 s on miniVite.

### Caveats

- Small structures are sampled at R too, so a single small buffer is either missed or over-weighted, and there is no per-object breakdown at sparse rates.

## Data

- **PolyBench:** 2mm, gemm, jacobi-2d and atax at MINI…MEDIUM (7 sizes), re-measured on 2026-10-08 with bytes touched.
  - Two splitter runs per size, 20 bins per interval: the 13 standard intervals (for the α model, Chao and the first hybrid), and intervals 95, 158, 253, 380, 791, 1582, 3164 and 7910, whose unions match the sampler runs' rates exactly.
  - Sampler runs with `-s 20 -r 20` (λ = 1, nearly independent bins) at `-i 25` and `-i 50` for every size, and at `-i 3, 5, 8, 12, 100, 250` for the held-out configs (MEDIUM, SMALL).
  - Snapshots every ~1/200 of each run; `-track_frees` on.
- **Evaluation:**
  - The largest (extrapolation) and the middle (interpolation) size are held out, as in the paper.
  - Error is MAPE over the snapshots of the held-out run, plus the relative error of the peak.
  - Sampler curves are compared on the fraction of their own run. For these single-threaded programs the sampler's estimated reference count is within 0.5% of the true count.

## What the tool now records

Each footprint keeps, per address, how often it was sampled, and reports f1..f4: the number of addresses sampled exactly 1, 2, 3 and 4 times, and how often an address entered the sample (`Discovered`). The splitter also keeps the union of each interval's 20 bins (timeline rows with `Bin = -2`), which is a 1-in-interval/20 sample. The sampler's main row is the union of its bins. Summary files are unchanged. With `-track_frees` the footprints count bytes touched; the frequency counts stay per start address.

## Results

Mean absolute error over the four kernels, in %, re-measured with bytes touched. Metrics:
- **MAPE** is over the snapshots of the held-out run.
- **Error of peak** compares the estimated maximum with the true maximum, wherever each occurs.
- **Error at peak** reads the estimate at the moment the true footprint peaks.

| Estimator | Sampler run | Extrap. MAPE / error of peak / error at peak | Interp. MAPE / error of peak / error at peak |
|---|---|---|---|
| α model per snapshot (paper features, NZ) | `-i 25`/`-i 50` | 43 / 27 / 61 | 31 / 59 / 44 |
| Chao1 | `-i 12` | 34 / 101 / 54 | 36 / 16 / 28 |
| iChao1 (adds f3, f4) | `-i 12`/`-i 25` | 34 / 107 / 55 | 45 / 17 / 31 |
| Known-rate (`richness.py`) | `-i 5` | 18 / 43 / 28 | 12 / 39 / 16 |
| **Hybrid** (known-rate × learned correction) | `-i 3` | **6 / 13 / 11** | **5 / 11 / 8** |
| Hybrid | `-i 25` | 25 / 20 / 32 | 12 / 23 / 23 |
| Hybrid | `-i 50` | 57 / 224 / 129 | 14 / 34 / 24 |

The sampler run for Chao and known-rate is the one closest to the union interval that fits the training sizes best; with the denser runs now available it is `-i 12` and `-i 5` (before: `-i 25` and `-i 3`). Under the paper's definition the rows were 45/30/58 | 31/59/49 (α), 28/9/33 | 57/39/60 (Chao1), 12/26/17 | 7/23/10 (known-rate), 6/11/12 | 5/9/8 (hybrid `-i 3`), 23/21/35 | 12/24/20 (`-i 25`) and 61/160/119 | 24/33/39 (`-i 50`).

From the splitter's own dense unions, known-rate alone gives 12% (extrapolation) and 7% (interpolation) MAPE; the α model gives 43% and 33%, and Chao1/iChao1 44–46% and 31–34%.

The **hybrid** regresses log(true / known-rate estimate) on the training sizes' splitter unions at the sampling rate closest to the run being predicted (Ridge, α = 1). Features:
- how far the known-rate estimate extrapolates beyond the observed footprint;
- log((f2+1)/(f1+1)), the fraction of samples that were new addresses, f1/seen, and the sampling rate;
- the bins' spread σ, mean observed footprint and interval (the α model's inputs);
- the share of new addresses among the last 5 snapshots' samples (`Discovered`).

Per kernel, hybrid, MAPE / error of peak:

| Kernel | `-i 3` extrap. | `-i 3` interp. | `-i 25` extrap. | `-i 25` interp. |
|---|---|---|---|---|
| 2mm | 5 / +8 | 8 / +17 | 21 / +10 | 17 / +38 |
| gemm | 3 / +15 | 6 / +11 | 19 / +3 | 7 / +10 |
| jacobi-2d | 4 / +6 | 3 / +8 | 38 / −27 | 11 / +24 |
| atax | 11 / +23 | 3 / +8 | 21 / +39 | 14 / +19 |

**Correction to earlier numbers:** a sampler's union holds only the sampled references that land in at least one bin, a fraction 1 − e^−λ of them, so its rate is (1 − e^−λ)/i, not 1/i. The first known-rate sampler numbers on this branch used 1/i, which overstated the sampled fraction 1.6×. `timeline.union_rate` recovers λ from the bin rows.

## Improving the hybrid at `-i 25`

The steps were tested in the order below (re-measured with bytes touched; the discovery feature of step 4 is now always present, see the last row). Mean absolute error over the four kernels at `-i 25`, extrapolation / interpolation, %:

| Step | MAPE | Error of peak | Error at peak |
|---|---|---|---|
| Before: correction trained on the splitter union with the nearest rate (1/37 for a 1/40 sampler) | 24.6 / 12.4 | 19.9 / 22.7 | 32.2 / 23.2 |
| 2. Correction trained at exactly the sampler's rate (splitter `-intervals 95,158,253,380,791,1582,3164,7910`) | 24.1 / 11.2 | 24.0 / 22.0 | 38.5 / 25.3 |
| 2 + 1. Estimate made non-decreasing between frees (isotonic per segment) | 23.5 / 10.0 | 25.0 / 17.5 | 37.3 / 26.3 |
| 2 + 3. One reuse shape per run, fitted on the run's data-rich snapshots | 24.1 / 11.2 | 31.8 / 19.0 | 49.2 / 27.0 |
| 2 + 1 + 3 | 23.2 / 10.4 | 25.2 / 18.3 | 40.0 / 28.2 |
| 2 + 1 without step 4 (the share of new addresses among the last 5 snapshots' samples, from the `Discovered` counter) | 22.8 / 10.4 | 26.3 / 16.9 | 40.4 / 26.7 |

- **Step 2** matters when the old rate mismatch was large. At `-i 50` (1.6× off before), extrapolation MAPE drops from 57% to 32%. At `-i 25`, where the mismatch was 8%, it changes little.
- **Step 1** lowers interpolation error and is best at dense rates. At `-i 3` it gives 4.4 / 4.2% MAPE and 3.4 / 7.0% error of peak.
- **Step 3** does not help: the per-snapshot shape is no worse once the correction is applied.
- **Step 4** (the discovery feature) lowers interpolation MAPE (10.4 → 10.0) and the error at the peak (40.4 → 37.3 for extrapolation), but not extrapolation MAPE (22.8 → 23.5).
- **What remains is a size bias.** At `-i 25` the largest size is underestimated, mostly in the first half of the run: signed mean error 2mm −23% (−34% in the first half), gemm −9% (−13%), jacobi-2d −38% (−41%), atax −3% (−23%). Interpolation is almost unbiased.
  - The correction is learned on the smaller sizes, and at the same rate and phase the largest size's sample statistics fall outside that range.
  - None of the four steps addresses that.
- **Size-invariant features do not fix it either.** The bins' σ and mean (bytes) were replaced by the relative spread σ/mean and the bin/union footprint ratio, with smoothing and the discovery feature on, at `-i 25`:

  | Feature set | MAPE | Error of peak | Error at peak |
  |---|---|---|---|
  | Original (σ and bin mean in bytes) | 23.5 / 10.0 | 25.0 / 17.5 | 37.3 / 26.3 |
  | Size-invariant (ratios only) | 26.4 / 20.0 | 18.0 / 15.6 | 32.6 / 27.4 |

  - Per kernel, extrapolation MAPE improves for jacobi-2d (37.5 → 25.0) and atax (22.0 → 21.8) but worsens for 2mm (23.4 → 27.4) and gemm (11.0 → 31.5).
  - Interpolation gets worse, and at `-i 50`..`-i 250` it is unstable (39–78% interpolation MAPE, against 16–20%).
  - So the byte features carry useful information inside the training range and are not what causes the bias. The original set stays the default; the invariant one is `HYBRID_FEATURE_SETS["invariant"]`.
  - The bias more likely sits in the sample statistics themselves. In 2mm and gemm each element is reused about N times, so a larger input means more reuse per address. At the same sampling rate the largest size then shows more repeats than any training size did, whatever units the features use.
- **Remaining options:** train the correction on the larger sizes only (the tiniest ones may mislead it), constrain it to extrapolate monotonically in the size-related features, or add a feature for reuse per address relative to the known rate (e.g. the expected samples per address implied by the known-rate fit).

## The hybrid between `-i 3` and `-i 25`

Where the hybrid falls off, and what each density costs. Sampler runs `-s 20 -r 20` at `-i` 3, 5, 8, 12, 25, 50, 100 and 250 on the held-out sizes (MEDIUM, SMALL) of 2mm, gemm, jacobi-2d and atax; the correction is trained on splitter unions at exactly each sampler's rate. Variant: known-rate per snapshot, smoothed between frees, with the discovery feature. Mean absolute error over the four kernels, extrapolation / interpolation, %; time is MEDIUM, averaged over 2mm, gemm and jacobi-2d (3 runs each, one at a time on a quiet machine):

| `-i` | MAPE | Error of peak | Error at peak | Time, MEDIUM (2mm / gemm / jacobi-2d) | Of a full trace |
|---|---|---|---|---|---|
| 3 | 4.4 / 4.2 | 3.4 / 7.0 | 11.2 / 7.9 | 28.2 / 20.9 / 35.7 s | 40% |
| 5 | 6.4 / 6.1 | 6.0 / 10.0 | 17.5 / 14.5 | 18.7 / 14.6 / 23.1 s | 27% |
| 8 | 15.0 / 9.3 | 33.1 / 14.0 | 33.4 / 17.4 | 12.7 / 10.3 / 15.4 s | 18% |
| 12 | 24.3 / 11.5 | 64.9 / 22.2 | 34.3 / 19.4 | 10.3 / 8.4 / 12.2 s | 15% |
| 25 | 23.5 / 10.0 | 25.0 / 17.5 | 37.3 / 26.3 | 7.1 / 5.9 / 7.6 s | 10% |
| 50 | 30.6 / 15.6 | 40.6 / 13.6 | 60.0 / 22.8 | 5.2 / 4.6 / 5.5 s | 7% |
| 100 | 29.7 / 20.3 | 17.1 / 29.2 | 54.3 / 26.7 | — | — |
| 250 | 44.7 / 18.8 | 23.8 / 28.6 | 55.0 / 50.5 | — | — |

The full trace (splitter, 8 intervals) took 71.4 / 50.5 / 90.6 s; native runs take 0.05–0.09 s.

- **The hybrid holds to `-i 5` and falls off between 5 and 8.** At `-i 5` it is within 3–9% MAPE and −7% to +14% of the peak on every kernel and split, for 67% of the cost of `-i 3` and 27% of a full trace. At `-i 8` 2mm's extrapolation goes to 25% MAPE and +56% on the peak, and atax's peak to +44%.
- **Past the fall-off it overshoots the peak.** At `-i 12` 2mm extrapolation is +161% on the peak, worse than at `-i 25` (−13%). The correction then extrapolates beyond its training range.
- Per kernel, MAPE / error of peak:

  | Kernel | `-i 5` extrap. | `-i 5` interp. | `-i 8` extrap. | `-i 8` interp. |
  |---|---|---|---|---|
  | 2mm | 6 / +6 | 8 / +12 | 25 / +56 | 11 / +16 |
  | gemm | 5 / +6 | 3 / +7 | 11 / +16 | 12 / +13 |
  | jacobi-2d | 6 / +3 | 8 / −7 | 11 / +17 | 5 / −5 |
  | atax | 9 / +9 | 5 / +14 | 13 / +44 | 9 / +21 |

- Under the paper's definition the fall-off was the same: 3.6 / 4.9 at `-i 3`, 5.9 / 5.8 at `-i 5`, 14.2 / 9.5 at `-i 8`.

## What we learned

Items 2–4 were measured under the paper's definition with ad-hoc scripts and were not re-run; items 1, 5 and 6 are re-measured.

1. **A single bin cannot see reuse.** Each bin holds 0.1–10% of the footprint and almost never samples an address twice. So a model working from one bin, like the α model, can't tell new memory from memory being touched again: on a plateau its estimate keeps rising. It has 43% / 31% MAPE.
2. **Chao1/iChao1 fix plateaus but are biased elsewhere.**
   - jacobi-2d extrapolation falls from 42–52% to 4% (splitter union) and 9% (sampler).
   - Kernels with many addresses touched only once or a few times (initialisation writes) are underestimated by 30–60% at moderate density, because Chao is a lower bound when reuse is uneven.
   - At the densest union (1 in 5) Chao overshoots by +50–245%, because f2 is much smaller than f1.
   - The sampler shows exactly the same bias as splitter unions of the same density.
3. **Using the known rate is the key.**
   - The sampling rate is known exactly: sampled references / references executed. Together with the reference count T and the seen count S, it pins down the access-count distribution well once the sample has repeats.
   - Dense unions (1 in 5): bias +1% to −12%, against +53% to +245% for iChao1.
   - At 1 in 12: within about ±10–20%.
   - At 1 in 50 and sparser: −90% to +215%, because f2..f4 are too small to identify the reuse shape.
4. **The reuse shape does not transfer between sizes.** Fixing the negative-binomial shape from the training sizes and solving only from S, T and p gives 40–150% error. The fitted shapes are tiny (0.02–0.08): real reuse is a mix of touched-once memory and heavily reused data, not one smooth distribution.
5. **Cost.** A sampler run dense enough for the known-rate estimator (`-i 3`) takes 40% of a splitter run, and `-i 5` 27%, but `-i 3` is 5.5× a sparse `-i 50` run. MEDIUM timings, 2mm / gemm / jacobi-2d: `-i 3` 28.2 / 20.9 / 35.7 s, splitter 71.4 / 50.5 / 90.6 s, `-i 50` 5.2 / 4.6 / 5.5 s. Unlike the α model, the known-rate estimator needs no training.
6. **A learned correction makes known-rate usable at sparser rates.**
   - The known-rate estimator's error is systematic: it underestimates while addresses have not yet been reused, which is exactly when f2/f1 is low. A regression on the training sizes learns this.
   - With the dense sampler (`-i 3`), the corrected estimate is within 3–11% MAPE on all eight held-out runs; with `-i 5` within 3–9%.
   - With `-i 25` it gets 7–38%.
   - At `-i 50` it is unstable.
7. **The error of the peak and the error at the peak are different questions.** The estimators get the maximum memory requirement right well before they get it right at the moment it happens. Example (paper's definition): raw known-rate on 2mm MEDIUM is +1% on the peak value but −19% at the peak time, because the estimate only catches up when the program starts re-reading its arrays halfway through.

## miniVite (system OpenMPI, superseded)

Measured before miniVite was rebuilt against spack's OpenMPI 5.0.5, under the paper's definition, with `OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive` and the f1/f2-only build (Chao1 but not iChao1 or known-rate). Kept for the record; not re-run.

- **Stable run lengths:** with pinned, passive threads the sampler and splitter runs agree within 1% on references executed. Unpinned, OpenMP threads spin and runs differed by up to ±50%.
- **Footprint not representative:** ~175 MB at every size from 1024 to 8192 vertices, mostly MPI start-up memory of the system OpenMPI.
- **Results on this data:** α model 22–25% (extrapolation) and 14–21% (interpolation) from splitter bins, 25–33% from the sampler; Chao1 34–35% from splitter unions and +219–241% from the sampler.

## Open questions / next steps

- **Window switching cost:** opening or closing a window retranslates all code, which makes cc1 slower with windows than fully traced. With bytes touched, 1% watched (or none, where memory outside blocks is small) is as accurate as 5%, so far fewer windows would do; alternatively switch with Pin's per-trace versioning instead of flushing the code cache.
- **Windowless mode:** pages alone are within 8% of every peak tested except where memory lies outside blocks. A mode that drops windows entirely would cost little more than the allocator hooks.
- **Watching adds a few points on cc1 and Lua** (selected bytes outside blocks); find what is counted there.
- **Spatial error bars on miniVite** are too narrow in a few runs; check whether word selection correlates buckets with miniVite's layout.
- **Ground truth:** allocator accesses are excluded for glibc only. Other allocators (jemalloc, tcmalloc, pymalloc's arenas inside Python) are neither tracked as blocks nor excluded.
- **Windowed density** is moot with bytes touched. It still applies to `-footprint starts`.
- **Hybrid below `-i 5`:** the correction overshoots once it extrapolates past its training range (`-i 8` and sparser); constraining it might move the fall-off.
- **Generalisation:** the correction is trained per workload on its smaller sizes, like the paper's α model. It has not been tested across workloads, or on miniVite.
- **Reuse model:** a two-class mixture (touched-once plus reused) instead of one negative binomial might transfer between sizes and allow sparser sampling.
- **Forecasting** is unchanged from `main`: the peak is within about 10–20% for interpolation and unreliable for extrapolation.
