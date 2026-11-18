# Experiments for Chapters 2 to 4 (paper_rss/dissertation_plan.md, Section 7)

Fixed before the runs below. Data: `~/memprint-data/2026-11-05-diss-ch2/`. Scripts: `dissertation/run.sh` (runs),
`dissertation/score.py` (analysis). Pin tool: `pintool/memprint_trace.cpp` at this commit, which adds two knobs that
change nothing when off:
- `-exclude_alloc 1`: the allocator's own references (glibc's chunk headers and free lists) count as time but not
  as footprint, without tracking frees. Before, only `-track_frees` and `-window` turned this on, so the bytes
  retrace of `data/polybench-bytes` (splitter, no free tracking) still counted them.
- `-disjoint 1` (sampler): each sampled reference goes into exactly one of the `-s` bins, so the bins are disjoint
  1-in-(i·s) subsamples, distributed as the splitter's bins at interval i·s. Same instrumentation cost as the
  bootstrap.

Done already and not repeated: R2 (page-fault sampler), R3 (status sampler at 100 µs), R4 (VmHWM), R6
(`PYTHONMALLOC=malloc`), R7 (cc1's GC heap): `paper_rss/sampling/`, commit c95c540. R5 is in `tab:thp`, `tab:alloc`
and the inflation paragraph of paper_rss.

## Ch. 2: MemPrint

### M1. The TACO protocol on bytes touched (analysis)
`memprint build` (analysis/memprint/build.py, unchanged: NZ, MT and L2O; EXTRA = largest config, INTER = middle) on
three sets of `allData` tables, copied so the originals are not overwritten:
- `taco`: the published tables (`~/memory_estimator/tools/data`), to check that the pipeline reproduces the
  published 9.78% and 3.70%;
- `starts`: `data/polybench-starts` (expected identical to `taco`);
- `bytes`: `data/polybench-bytes` (PolyBench retraced with `-footprint bytes`; miniVite from that retrace, with the
  allocator's references still in, replaced by M4).

Reported: mean, median and maximum test MAPE at each subset's training-best interval (the published table's
basis), per split and subset, over the 28 workloads (27 PolyBench kernels and miniVite), and per workload for L2O.

Run during setup (the analysis takes a minute): `taco` gives L2O 9.72% (EXTRA) and 3.79% (INTER) against the
published 9.78% and 3.70%, with the published maxima exact (atax 46.83% L2O, trisolv 109.23% MT, 2mm 21.62% MT);
`bytes` gives 11.91% and 3.41%, with bicg at 71.8%. The 0.06 and 0.09 point residual of the reproduction is to be
explained (jacobi-1d and durbin are in the published tables but not in `analysis/workloads.txt`, and the published
pipeline rounded coefficients to 4 decimals).

### M2. The sampler's bin interval and rate (analysis)
1. Read the published sampler (`buffer_memtrace_poissonBS.cpp` in Pin's ManualExamples, the one built there).
   Found during setup: it draws K ~ Poisson(λ) with λ = s/r and puts the reference into K distinct bins out of s,
   so a bin's inclusion probability is E[min(K, s)]/s ≈ 1/r and its interval about i·r. It labels the bins
   `SubSample_<i·s/r>` = i·λ. The label equals the bin interval only when s = r² (the defaults s=100, r=10). The
   paper's worked example (i=1000, 50 bins, r=20) would be labelled 2,500 while each bin samples about 1 in 20,000.
   It also stores addresses as `UINT32` (truncated to 32 bits), and it does not cap K at s.
2. Find the sampler runs behind the published predictions and overheads (`~/memory_estimator/tools/traces/*-old`,
   `*-recentOld` and others: file names give i and the label), recover their (i, s, r), and measure each bin's
   empirical interval as the run's reference count over the bin's `CountObs`.
3. Find the code that matched sampler bins to a trained model's interval, and whether it used the label.

Outcome: a table of (i, s, r, label, empirical interval) per run set, and whether any published prediction used a
mislabelled interval. If one did, M3's runs replace it.

### M3. Sampler bins against splitter bins (Pin, PolyBench minutes per run)
The published test MAPE (`build`'s test columns) is computed on the splitter's bins at the held-out config, not on
sampler output, so it assumes that a sampler's bins are distributed like the splitter's. Under the bootstrap a
reference that is replicated enters several bins, so the bins are positively correlated, and the spread across them
(`SD_MemUsage`, a model feature) should be smaller than across disjoint splitter bins. Expected: SD ratio < 1.

Runs, current tool, `-footprint bytes`, one at a time:
- Workloads and configs: the 27 PolyBench kernels at their EXTRA and INTER configs; miniVite at 16384 and 4096
  (after M4, with `-exclude_alloc 1`, `OMP_NUM_THREADS=1`).
- Intervals: for each workload and split, every interval that a fitted subset uses for prediction in the `bytes`
  fits (each subset's training-best interval and the best monotone interval), from M1 (M4 for miniVite).
- Samplers at bin interval k: bootstrap with the published defaults s=100, r=10, i=k/10 (λ=10, bin interval
  ≈ k); disjoint with s=20, i=k/20 (bin interval exactly k, 20 bins as the splitter).
- 3 seeds each (`-seed 1,2,3`).

Scored: per run, the bins' mean `MemUsageObs` and `SD_MemUsage` over the splitter's at the same k and config; the
α MAPE of the workload's fitted model (each subset) on the sampler's bins against its MAPE on the splitter's bins.

Decision rule: if the bootstrap's α MAPE is more than 2 points above the splitter-bin MAPE at the median over
workloads, or its median SD ratio is outside 0.9 to 1.1, the chapter reports predictions from the disjoint sampler
(and states the bootstrap result); otherwise it reports that the bootstrap is equivalent, with the numbers.

### M4. miniVite retraced under the corrected definition (Pin, about 2 h)
`scripts/run.sh minivite --mode splitter --footprint bytes --exclude-alloc --runs 1`, configs 1024 to 16384,
`OMP_NUM_THREADS=1` (the thread count of the bytes retrace), then `memprint preprocess` and `build`, and M3's
miniVite runs. Reported: the truths against the start-address and the bytes-with-allocator truths, and the six test
MAPEs against the published (NZ 3.98%, MT 4.98%, L2O 12.90% EXTRA; 2.94%, 1.04%, 1.57% INTER).

### M5. The kernels missing from the bytes tables (Pin, PolyBench)
durbin and jacobi-1d are in the published tables (jacobi-1d in training only) but were not retraced. Splitter
`-footprint bytes`, 7 configs each, then M1's analysis with them. If either fails to trace, the chapter's table
states the 27 kernels covered.

### M6. Massif and DHAT against the corrected truth (Valgrind, native)
Massif (peak heap) and DHAT (`tgmax`) on miniVite at 1024 to 16384 (`OMP_NUM_THREADS=1`) and on gemm at its 7
configs (the published Figure 1 workloads), 1 run each (both are deterministic for these inputs). Compared with the
bytes truth (M4 for miniVite, `data/polybench-bytes` for gemm) and with MemPrint's L2O estimate from M3's disjoint
or bootstrap runs, per the M3 decision. Expected from paper_sched: the heap profilers overestimate miniVite under
bytes touched (1.42× to 1.55× at 32,768 and 65,536 vertices), the opposite of the published direction.

### M7. Sampler overhead on one idle machine (Pin, native)
miniVite at 8192 vertices, `OMP_NUM_THREADS=1`, 5 runs each, interleaved, nothing else running: native;
instrumentation only (`--mode instr`); the published sampler (`buffer_memtrace_poissonBS.so`) at i = 500, 1000 and
2500 with s=50, r=20; the current tool's bootstrap and disjoint samplers at the same i; the splitter. Reported:
median wall time per configuration, against the published 52.2 s (1 in 1000), 46.1 s (1 in 2500) and 42.1 s
(1 in 500). Run last, after every other run.

### M8. Bins, intervals and training overhead (analysis)
From the published tables: rows per config and interval (bins), the intervals present, and the intervals each
workload drops (`dataset.prepare`: an interval with an empty bin). Found during setup: 20 bins per interval and 13
intervals (100 to 100,000) in every table, with interval 10 absent. The cumulative overhead in
`~/memory_estimator/tools/plot_cumulative_accuracy.py` is 216.375, 483.772, 719.736, 1,093.118 s, so the chapter's
438.77 s is a transposition of 483.77. The overhead values are hardcoded in the plot scripts; M7 re-measures them.

## Ch. 3

### R1. The resident-memory model on MemPrint's inputs (native)
Each of the 27 PolyBench kernels at its 7 configs and miniVite at its 5 (`OMP_NUM_THREADS=1`), 3 runs, under the
shim's `pss` mode every 1 ms (`MPR_INT=0.001`, the sampler that paper_rss/sampling found best for short runs).
Truth: the bytes truth of the same config (`data/polybench-bytes`; M4 for miniVite). Both quantities count the
whole run, and the PolyBench kernels allocate their arrays once, so the peak of memory in use equals the bytes
touched up to the runtime's share.

Scored as the blind test of paper_rss (`paper_rss/blind/score.py`): (a) the largest A − F of the run, median of 3;
(b) (a) plus R, the median of truth − measured over the other workloads' configs. Reported at the EXTRA and INTER
configs (the M1 splits), as absolute % error, next to MemPrint's test MAPE there (M1 for the splitter bins, M3 for
the sampler) and the cost of each (one native run against one sampled Pin run). Secondary: the transfer rule of
paper_rss from the smaller configs to EXTRA, with the input size from the PolyBench dataset parameters.

## Ch. 4 (priority 3, design only until Ch. 2 and 3 are done)

### L1. Realloc chains and container growth at native speed
libmpr's profiler records a realloc chain as one row where Pin records every buffer. The MemPrintResize pass already
calls `__mpr_growth` and `__mpr_capacity` at container growth; libmpr would log (site, old size, new size, chain id)
at each, and the profile would be compared with Pin's `-sites` profile on the programs of paper_resize that grow
containers (per-site requested and touched bytes, commit).

### L2. Small blocks without Pin
A compiler pass that instruments loads and stores into a per-page shadow (one bit per 4 KiB page per block of less
than 64 KiB, as sanitizers keep shadow memory), giving per-site touched pages at compile-time cost. Compared with
Pin's `-sites` profiles on the same programs: profile agreement per site and run time.

## Amendments after the runs (2026-11-06)

Scope (user): Ch. 2 and Ch. 3 as above; Ch. 5 and 6 skipped. All Ch. 2 numbers use bytes touched as the truth, and
M7's measurements replace the published overheads (same machine).

### M2 found (scan: `m2_runs.csv`, one row per sampler run, empirical interval = i x run references / bin `CountObs`)
- Every table behind the published predictions (`tools/traces/<workload>`, read by `tools/preprocess_fixed_split.py`)
  is splitter output, labelled exactly. No published prediction used sampler bins.
- The published sampler's runs are only in `workloads/miniVite/traces` (`Sampled_*`): the 50-bin runs (s=50, r=5)
  are labelled i x 10 and sample 1 in i x 5 (label 2x the bin interval); the 100-bin runs (s=100, r=10) are exact.
  Older tools' runs (`Buffered_*` with bins) are exact in gemm_prev, mvt-1000 and correlation-recentOld and labelled
  5x to 10x too small in correlation-old and miniVite_fileInput; none of them feeds a published number.
- The published overheads are the s=50, r=5 runs of `overhead_sampler_miniVite_SI.csv`, plotted by
  `tools/plot_PIN_overhead.py` at `SI x 10` (the label). The chapter's "42.1 s at 1 in 500" is i=50 at 8192 vertices
  (42.12 s), whose bins sample about 1 in 250 and whose instrumentation fires on 1 in 50 references. 52.2 and 46.1
  are not in these logs.

### M7b (added): overheads at matched bin intervals
M7 ran i = 500, 1000, 2500 as the plan states, which is not what the published numbers measured. `run.sh m7b`
times each sampler at bin interval k = 500, 1000, 2500: the published sampler with s=100, r=10, i=k/10 (exact
label) and with the published s=50, r=5, i=k/5; the current bootstrap (s=100, r=10, i=k/10) and disjoint (s=20,
i=k/20); native; 5 interleaved rounds. Reported: median per configuration (`m7_k.csv`).

### M9, R8 and R9 (added Nov 6, after the plan's update)

- M9 (`run.sh m9`, `score.py m9`): the truth run is the splitter with one interval and one bin (`-intervals 1 -bins 1`),
  which gives the splitter's own interval-1 footprint (gemm SMALL: 222,014 against 222,010 B). Five workloads, six runs
  each (native, instrumentation only, truth, bootstrap and disjoint at 1 in 1,000, splitter), 5 interleaved rounds.
- R8 (`run.sh r8`, `score.py r8`; paper_rss/sampling `SAMPLERS="pss1 event" LARGE=large_r8 run.sh large`,
  `SAMPLERS=event BLIND=event run.sh small blind`, `score.py peaks transfer cost_r8 blind`): `MPR_MODE=event` of
  mprshim.c built with `-DMPR_EVENT_FREE`. Choices fixed before the runs: the free threshold is 4 KiB, not a large
  size, because any free after the peak lowers anon - free; a free or munmap triggers a sample only if the process has
  faulted since the last one; the sample reads smaps_rollup with open/read so that it allocates nothing.
- R9 (`r9pages.sh`, `r9score.py`): trisolv's +80% led to a pages-touched truth for every R1 test input. The pages
  truth is cumulative distinct pages, not live pages, so it overstates programs that free and reuse (miniVite).
- R8, second run (reported): the first run sampled before every free >= 4 KiB after any fault and slowed sqlite
  1.99x. Added before the second run: a free triggers a sample only if MPR_GAP (1 ms) or ten times the last sample's
  duration has passed, or the process has faulted in MPR_GROW (1%) of its anonymous bytes since the last sample.
  Every R8 measurement was repeated with this rule.
- R1 is scored primarily against the pages-touched truth (`score.py r1p`, r9pages.sh now covers all 194 inputs),
  with the bytes-touched truth as the second column.

### M10, M11, R10 and R11 (added Nov 6, after the plan's second update)

- M10 (`m10.py`): the 56 selected models of the published `{extra,inter}polation_models.csv` against the refitted
  `$OUT/m1/taco/data/models.csv`, coefficient by coefficient at 4 decimals and by interval. The published
  per-workload L2O errors are not stored, so the comparison stops at the models.
- M11 (`m11.sh trace`, `score.py m11build`, `m11.sh sample`, `score.py m11`): the splitter with `-footprint pages`
  (default 13 intervals x 20 bins; miniVite with `-exclude_alloc 1`) on every PolyBench and miniVite input, 12 at a
  time, binaries built first. The tool counts pages only in the exact footprint (Bin -1), and the bins stay in bytes,
  so M11 is M1 with the pages truth. The sampler runs use `-footprint bytes` (the tool rejects pages in sampler mode,
  and Pin exits 0 on that usage error: the first sampler run produced no bins and was discarded). The run list comes
  from the pages fits (172 runs per sampler and seed).
- R10 (`score.py r10`): the in-run base is the first sample of each event (and 1 ms) run: anon - free (`base0`) and
  RSS - anon (`file0`, Chapter 3's). (b) is measured + base + a constant (the median of truth - measured - base over
  the other workloads); (c) scales measured - base and adds the target's base. The blind NPB configurations are
  rescored with the same transfer (paper_rss/blind/score.py `transfer(..., base="base0")`).
- R11 (`m11.sh trace`, `score.py r11`): miniVite's pages in use, `-mode splitter -intervals 1 -bins 1 -footprint pages
  -track_frees 1 -snapshot 100000`, the largest Bin -1 value of the timeline; R1 rescored with it as miniVite's truth.
- M7 splitter check (added Nov 9): the published splitter (`buffer_memtrace_poisson_cascade.so -short_name`) once
  at 8,192 vertices on the current miniVite binary, with `OMP_NUM_THREADS=1` (113.0 s) and unset (32 threads,
  386.9 s; published 373.6 s from `runtime_miniVite.sh`, which never sets it). The tool writes into `./traces/`.

### B50 (added Nov 11): 50 splitter bins per interval
`bins50.sh trace` (splitter `-bins 50`, bytes, 13 default intervals; miniVite `-exclude_alloc 1`, its five inputs alone
and in sequence for the cumulative cost; PolyBench 12 at a time), `score.py b50build` (preprocess, build, run list
from the new fits: 137 runs per sampler and seed), `bins50.sh sample` (bootstrap s=100, r=10, i=k/10 as M3; disjoint
s=50, i=k/50), `score.py b50`, `bins50.sh time` (splitter at 20 and 50 bins on miniVite 8,192, 5 interleaved
rounds). Checked first: 50 bins needs 2.9 GB on miniVite 16,384 and 3.9 GB on lu MEDIUM (20 bins: 1.5 and 1.6 GB).
Data: ~/memprint-data/2026-11-11-bins50.
- B50 sampler cost (added Nov 11, `bins50.sh samptime`): miniVite 8,192 at k = 500, 1,000, 2,500 (M7b's default
  knobs) for bootstrap (s=100, r=10), disjoint 20 and disjoint 50 bins, plus native; gemm, 2mm, jacobi-2d MEDIUM at
  k = 1,000 with `-footprint bytes` next to the truth run (M9's); 5 interleaved rounds, nothing else running.

### B50 follow-ups (added Nov 12)
`batch50.sh trace` (12 at a time): durbin and jacobi-1d at 50 bins (m5); M11 at 50 bins, `-footprint pages` (p50); the
corpus programs with an input size at 6 or 7 inputs (GAP bfs/cc/pr/sssp -g 12..17, HPCCG 16..48, LULESH -s 10..30
-i 20, MiniFE 16..40), bytes, 50 bins (c50). NPB (fixed classes, only S/W/A traceable) and darknet (one network) left
out. Four corpus traces were OOM-killed at 12 at a time (12 to 33 GB each) and rerun 3 at a time
(`batch50_retry.sh`). Then `score.py b50more_build`, `batch50.sh sample` (bootstrap s=100 r=10; disjoint 50 bins),
`score.py b50more`; alone: `batch50.sh cost` (native, instr, truth and the three samplers at 1 in 1,000 in the same
5 rounds), `batch50.sh splittime` (50-bin splitter per input, one at a time), `buildtime.py` (preprocess and build,
cProfile, 3 repetitions). Chain: `batch50_chain.sh`. Data: ~/memprint-data/2026-11-12-b50more.

### Corpus follow-ups and profiler checks (added Nov 13 to 16)
`corpusmore.sh` (chain `corpusmore_chain.sh`; traces under `memsched.py`, 48 GB): the corpus programs with
`-footprint pages` at 50 bins (cp50) and bytes at 20 bins (c20), refits (`score.py corpusmore_build`), both samplers
at the held-out inputs (and durbin/jacobi-1d's at 50 bins), `score.py corpusmore`, the 50-bin splitter per corpus
input alone, `buildtime.py corpus` (35 workloads). Three bootstrap runs on GAP pr 17 were OOM-killed (sampler jobs
budgeted at 2 GB, they need 12) and rerun alone (`corpusmore.sh retry`). Data: ~/memprint-data/2026-11-13-corpusmore.
Decision-rule check: bootstrap SD ratio on runs matched in workload, input, interval and seed between m3_runs.csv
and m3_runs_b50.csv. Profiler checks: Lua (input 1) under Massif with --heap-admin=0 and 8, DHAT's blocks at t-gmax,
the bytes truth with the current tool under glibc, jemalloc and TCMalloc and with the October 11 tool
(2026-10-11-sched/corpus/tool.so); truth over Massif's requested heap at the peak (corpus, largest inputs); the largest
gap between reference and wall-clock progress over 20 windows from nativerefs.

### Corpus truths with the current tool (added Nov 17)
The 56 scheduling-corpus truths (bytes, 2026-10-13-sched-llm/corpus; pages, 2026-10-12-resize/corpus-pages and
2026-10-26-rsscurve/llm) traced again with the frozen current tool through `scripts/footprint_corpus.sh` under
`memsched.py` (48 GB, 12 jobs), native dirs linked from sched-llm; compared by `truthcur.py` (peak and area of the
resampled curves) -> ~/memprint-data/2026-11-17-truthcur/compare.csv. Lua's bytes truth repeated three more times with
the current tool (scratchpad) after its first check (53.56 MB) disagreed with the corpus rerun (57.10 MB).

### Modeling statements (added Nov 17)
`modelcheck.py` on the 50-bin splitter bins of the 35 workloads (B50 data and b50more/c50 data), no new traces:
squared and cubic terms, a pooled leave-one-workload-out model (with and without the target's training inputs),
the slope of log alpha on log k per input, the rank of the best monotone interval with training-only selection, and
L2O with training-only selection (`modelcheck.py l2o_trainsel`). Outputs: $OUT/modelcheck_{terms,slope,rank,l2o_trainsel}.csv.

### Selection on the training inputs (added Nov 17)
`memprint build` now selects on the training inputs of each split (`--select-on train`, the default): the empty-bin
pruning, the monotone scores and the best interval come from `prepare` on the training configs; test rows are the
held-out config's bins at the same intervals, empty ones dropped and counted (`test_dropped`, 0 everywhere);
`select_models` picks NZ or L2O by training MAPE at the best interval (`--select-on all` keeps the published
selection and the test-MAPE choice; `score.py m1` uses it for the TACO-appendix sets). Rebuilt: m1/final, m1/bytes-m4,
m11, bins50, b50more/{c50,m5,p50}, corpusmore/{c20,cp50}; old fits kept as fits_all.csv / models_all.csv, old scores
in $OUT/selall/. Sampler runs at the newly needed intervals: `samplefill.sh $OUT/samplefill_missing.csv` (172 rows x
3 seeds = 516 runs, memsched, the set's own knobs and done format). Rescored with `score.py m3 m11 b50 b50more
corpusmore r1 r8 r1p r11`; `modelcheck.py` (terms and pooled now on training-only selection); `matched.py`;
`tables.py m1 l2o m3 m4 b50 b50_workloads unified b50_corpus m11 r1 r1p trainsel`; `figures.py sd m3 cumulative`.

### Motivation checks (added Nov 18)
`mp_errors.py` (MemPrint's signed L2O EXTRA errors on the 35 workloads: mean of per-bin estimates / truth - 1, per
sampler and seed) and `motiv_checks.py` (DMSched-Sim on the 2026-10-13-sched-llm corpus, synthetic stream, load 0.8,
seeds 0-2, both truths): e1 oracle memprint-peak (true peak x (1+e), margin = q90 of the other workloads' needed
margins) for boot/disj/split errors and every margin policy, e1cap its memory sweep; e2 request = 1.5/2/3 x peak
(real.make_jobs overestimate) and the MetaCentrum2 stream (kind=trace); e3 fixed parts unscaled (real.fixed_parts:
intercept of peak on INPUT_SIZE, clipped to [0, smallest-input peak]; run.Setup.required on the curves without them),
e3cap its sweep. Data ~/memprint-data/2026-11-18-motivchecks. The MetaCentrum2 grid lost 3 tasks and 1 run to memory
(28 workers each parsing the SWF log); those 4 were rerun with 4 workers.
