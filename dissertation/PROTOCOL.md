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
