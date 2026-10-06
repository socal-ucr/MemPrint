# Findings: footprint over time (branch `timeline-chao`)

Goal: reconstruct a program's live memory footprint over time from a sparse Pin trace. The data is the timelines written with `-snapshot`/`-track_frees` (see README).

On `main`, the paper's α model applied per snapshot gives 25–55% MAPE. This branch adds per-address sample counts to the Pin tool and tries estimators of the memory that was never sampled.

## Data

- **PolyBench:** 2mm, gemm, jacobi-2d and atax at MINI…MEDIUM (7 sizes).
  - One splitter run per size, using the 13 standard intervals with 20 bins each.
  - Sampler runs with `-s 20 -r 20` (λ = 1, nearly independent bins) at `-i 25` and `-i 50`.
  - `-i 3` runs for the held-out configs (MEDIUM, SMALL).
  - Snapshots every ~1/200 of each run; `-track_frees` on.
- **Evaluation:**
  - The largest (extrapolation) and the middle (interpolation) size are held out, as in the paper.
  - Error is MAPE over the snapshots of the held-out run, plus the relative error of the peak.
  - Sampler curves are compared on the fraction of their own run. For these single-threaded programs the sampler's estimated reference count is within 0.5% of the true count.

## What the tool now records

Each footprint keeps, per address, how often it was sampled, and reports f1..f4: the number of addresses sampled exactly 1, 2, 3 and 4 times. The splitter also keeps the union of each interval's 20 bins (timeline rows with `Bin = -2`), which is a 1-in-interval/20 sample. The sampler's main row is the union of its bins. Summary files are unchanged.

## Results

Mean absolute error over the four kernels, in %:

| Estimator (on the union of bins) | Extrap. splitter MAPE | Extrap. sampler MAPE / peak | Interp. splitter MAPE | Interp. sampler MAPE / peak |
|---|---|---|---|---|
| α model per snapshot (paper features, NZ) | 42 | 45 / 30 | 31 | 31 / 59 |
| Chao1 | 38 | 28 / 9 | 30 | 57 / 39 |
| iChao1 (adds f3, f4) | 40 | 27 / 11 | 29 | 55 / 36 |
| **Known-rate** (`analysis/memprint/richness.py`) | **12** | **17 / 5** | **7** | **20 / 8** |

How each sampler column was produced:
- **Splitter MAPE:** each estimator uses the union interval with the lowest error on the training sizes, which is the densest union, 1 in 5, for all of them.
- **Known-rate sampler columns:** the `-i 3` run (union ≈ 1 in 5).
- **Chao1 / iChao1 sampler columns:** their closest sparse run (`-i 25`).

Per kernel, known-rate, sampler `-i 3`, MAPE / peak error:

| Kernel | Extrapolation | Interpolation |
|---|---|---|
| 2mm | 18 / +1 | 19 / −5 |
| gemm | 18 / −3 | 19 / −11 |
| jacobi-2d | 5 / +15 | 12 / −7 |
| atax | 27 / 0 | 28 / −11 |

## What we learned

1. **A single bin cannot see reuse.** Each bin holds 0.1–10% of the footprint and almost never samples an address twice. So a model working from one bin, like the α model, can't tell new memory from memory being touched again: on a plateau its estimate keeps rising.
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
5. **Cost.** A sampler run dense enough for the known-rate estimator (`-i 3`) takes about 40–55% of a splitter run, but about 5× a sparse `-i 50` run. MEDIUM timings: 2mm 24.7 s vs 58.3 s (splitter) vs 5.4 s (`-i 50`); gemm 19.8 vs 43.5 vs 4.8; jacobi-2d 31.9 vs 67.4 vs 5.3. These were measured with other runs in parallel, so they are rough. Unlike the α model, the estimator needs no training.

## miniVite

Re-measured with `OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive`, using the f1/f2-only build, so Chao1 but not iChao1 or known-rate.

- **Stable run lengths:** with pinned, passive threads the sampler and splitter runs now agree within 1% on references executed. Unpinned, OpenMP threads spin and runs differed by up to ±50%.
- **Footprint not representative:** it is ~175 MB at every size from 1024 to 8192 vertices, against 7–14 MB in the paper's runs. The likely cause is that the system OpenMPI (`mpi/openmpi-x86_64`) touches large buffers at start-up; the paper used spack's openmpi-5.0.5. Until miniVite is built against that MPI, these numbers mostly measure MPI start-up memory.
- **Results on this data:**
  - α model: 22–25% (extrapolation) and 14–21% (interpolation) from splitter bins; 25–33% from the sampler.
  - Chao1: 34–35% from splitter unions; +219–241% from the sampler (the sparse-union overshoot).

## Open questions / next steps

- **Cost vs density:** the known-rate estimator needs about 1-in-5 to 1-in-12 sampling of references. Measure where accuracy falls off (e.g. `-i 3, 5, 8`) and whether a hybrid helps: known-rate early in a run, when footprints grow and samples are dense relative to them, and α or Chao later.
- **Reuse model:** a two-class mixture (touched-once plus reused) instead of one negative binomial might transfer between sizes and allow sparser sampling.
- **miniVite:** build against spack openmpi-5.0.5 and re-measure with all three estimators.
- **Forecasting** is unchanged from `main`: the peak is within about 10–20% for interpolation and unreliable for extrapolation.
