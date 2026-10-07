# Findings: footprint over time (branch `timeline-chao`)

Summary: sampling *addresses* instead of references (`-mode spatial`) estimates the live footprint over time within about 1–10% at 1-in-25 to 1-in-250 addresses on PolyBench, and within 0.2–6% on miniVite with 1, 4 or 16 threads. It needs no training and runs at about the cost of instrumentation alone (miniVite 8192: ~30 s vs ~180 s for a full trace). Everything after "Data" documents the reference-sampling route that led there.

Goal: reconstruct a program's live memory footprint over time from a sparse Pin trace. The data is the timelines written with `-snapshot`/`-track_frees` (see README).

On `main`, the paper's α model applied per snapshot gives 25–55% MAPE. This branch adds per-address sample counts to the Pin tool and tries estimators of the memory that was never sampled.

## Headline: spatial (address) sampling

`-mode spatial -i R -s 20` selects 1-in-R *addresses* with a salted hash and records every access to them. An address is then in the sample with probability 1/R whatever its access pattern, so the footprint estimate is simply the selected footprint × R, with no model and no training. The 20 hash buckets give each snapshot a standard error.

This removes the core difficulty of all the reference-sampling estimators below. With reference sampling, an address accessed r times is seen with probability 1 − (1 − p)^r, and r is unknown and grows with input size.

Selected accesses are rare, so they are recorded immediately, under a lock, as they happen; frees are applied as they happen too. The order across threads is therefore exact, and time is the true count of references executed.

### PolyBench

Held-out runs (largest and middle size of 2mm, gemm, jacobi-2d, atax; truth from the splitter); mean absolute error, extrapolation / interpolation, %:

| Method | MAPE | Error of peak | Error at peak | Runtime, MEDIUM (2mm / gemm / jacobi-2d) |
|---|---|---|---|---|
| Spatial 1/25 addresses | 0.9 / 3.4 | 0.8 / 3.4 | 0.8 / 3.4 | 6.0 / 3.9 / 4.2 s |
| Spatial 1/100 | 2.3 / 4.2 | 2.3 / 3.9 | 2.3 / 4.0 | 3.7 / 3.3 / 3.4 s |
| Spatial 1/250 | 2.3 / 9.9 | 2.4 / 9.4 | 2.3 / 9.6 | 3.5 / 3.4 / 3.3 s |
| Spatial 1/1000 | 6.7 / 17.5 | 6.6 / 14.9 | 6.6 / 27.2 | 3.3 / 3.2 / 3.1 s |
| Best reference-sampling hybrid, `-i 25` | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 | 7.6 / 6.5 / 8.2 s |
| Best reference-sampling hybrid, `-i 3` | 4.8 / 4.2 | 6.2 / 6.4 | 9.1 / 9.4 | ~25 / 20 / 32 s |
| Splitter (full trace) | — | — | — | 41.3 / 31.8 / 47.9 s |

- **Accuracy follows the binomial prediction,** relative error ≈ 1/√(addresses / R).
  - The largest sizes (1–1.6 MB, about 130–200k addresses) stay within 1–7% down to 1/1000.
  - The middle sizes (about 0.22 MB, about 28k addresses) need 1/100 or denser. At 1/1000 only about 28 addresses are selected.
  - The error is largest early in a run, when the footprint is still tiny.
- **The bucket error bar is honest at dense rates.** The truth lies within ±2 standard errors in 100% of snapshots at 1/25 and 1/100, 88% at 1/250 and 77% at 1/1000 (nominal 95%; a 20-bucket standard error is itself noisy).

### miniVite and multithreading

miniVite was rebuilt against spack's OpenMPI 5.0.5, the MPI behind the paper's runs, and run with `OMP_WAIT_POLICY=passive`.
- **Footprint:** a full trace of 1024 vertices gives 6.8–7.0 MB cumulative footprint, matching the paper's 6.84 MB. With the system OpenMPI it was ~175 MB, mostly MPI start-up memory.
- **References:** 44 million instead of the paper-era 1.76 billion. About 97% of those were OpenMP threads spinning while they waited.

Every spatial run is compared with the full trace of the same size and the same thread count, with no training. The live peak with `-track_frees` is 4.9, 6.2 and 7.9 MB at 1024, 4096 and 8192 vertices, the same at every thread count.

| Threads | Spatial 1/25 | 1/100 | 1/250 | 1/1000 |
|---|---|---|---|---|
| 1 | 0.2–0.9% | 0.4–1.0% | 2.0–3.3% | 1.9–4.1% |
| 4 | 0.2–1.3% | 1.0–1.6% | 1.2–4.9% | 1.3–6.0% |
| 16 | 0.4–0.8% | 0.6–2.0% | 1.3–2.3% | 1.5–6.1% |

(MAPE range over the three sizes.)

- **Peaks and error bars hold under threads.** Error of peak is ≤ 2.8% in every case. The truth lies within ±2 standard errors in 94–100% of snapshots, and run lengths match the full trace exactly.
- **Cost, 8192 vertices:**

  | Threads | Native | Splitter | Spatial (1/25 … 1/1000) |
  |---|---|---|---|
  | 1 | 0.24 s | 168 s | 28–32 s |
  | 4 | 0.20 s | 198 s | 28–41 s |
  | 16 | 0.28 s | 176 s | 26–38 s |

  The per-access lock is not a bottleneck at these rates.
- **Multithreading bugs found and fixed.** `tests/pintool/parallel.c` has 8 threads with 12 MB live at a barrier.
  - Buffered modes kept each thread's accesses in its own trace buffer, so a free in one thread could be applied before another thread's earlier accesses: the full-trace peak came out at 9.8–10.8 MB.
  - With `-snapshot`/`-track_frees` the buffer is now 16 pages, giving 12.08 MB on every run; default outputs are unchanged.
  - A first spatial fix held frees back with vector clocks. It was correct, but on miniVite, which frees constantly while MPI helper threads rarely flush, it took 7474 s. Recording spatial accesses immediately replaced it: 12 runs give 12.03–12.27 MB, with 28–40 s on miniVite.

### Caveats

- Small structures are sampled at R too, so a single small buffer is either missed or over-weighted, and there is no per-object breakdown at sparse rates.
- Hashing start addresses mirrors the paper's footprint definition (largest access size per start address).

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

Mean absolute error over the four kernels, in %. Metrics:
- **MAPE** is over the snapshots of the held-out run.
- **Error of peak** compares the estimated maximum with the true maximum, wherever each occurs.
- **Error at peak** reads the estimate at the moment the true footprint peaks.

| Estimator | Sampler run | Extrap. MAPE / error of peak / error at peak | Interp. MAPE / error of peak / error at peak |
|---|---|---|---|
| α model per snapshot (paper features, NZ) | `-i 25`/`-i 50` | 45 / 30 / 58 | 31 / 59 / 49 |
| Chao1 | `-i 25` | 28 / 9 / 33 | 57 / 39 / 60 |
| iChao1 (adds f3, f4) | `-i 25` | 27 / 11 / 33 | 55 / 36 / 58 |
| Known-rate (`richness.py`) | `-i 3` | 12 / 26 / 17 | 7 / 23 / 10 |
| **Hybrid** (known-rate × learned correction) | `-i 3` | **6 / 11 / 12** | **5 / 9 / 8** |
| Hybrid | `-i 25` | 23 / 21 / 35 | 12 / 24 / 20 |
| Hybrid | `-i 50` | 61 / 160 / 119 | 24 / 33 / 39 |

From the splitter's own dense union (1 in 5), known-rate alone gives 12% (extrapolation) and 7% (interpolation) MAPE; the α model gives 42% and 31%, and Chao1/iChao1 38–40% and 29–30%.

The **hybrid** regresses log(true / known-rate estimate) on the training sizes' splitter unions at the sampling rate closest to the run being predicted (Ridge, α = 1). Features:
- how far the known-rate estimate extrapolates beyond the observed footprint;
- log((f2+1)/(f1+1)), the fraction of samples that were new addresses, f1/seen, and the sampling rate;
- the bins' spread σ, mean observed footprint and interval (the α model's inputs).

Per kernel, hybrid, MAPE / error of peak:

| Kernel | `-i 3` extrap. | `-i 3` interp. | `-i 25` extrap. | `-i 25` interp. |
|---|---|---|---|---|
| 2mm | 6 / +9 | 6 / +15 | 20 / +20 | 18 / +39 |
| gemm | 3 / +9 | 5 / +9 | 16 / +3 | 8 / +16 |
| jacobi-2d | 5 / +7 | 2 / +4 | 31 / −20 | 10 / +21 |
| atax | 11 / +19 | 5 / +8 | 24 / +40 | 11 / +19 |

jacobi-2d's error at peak stays at −15% to −25% at every density, because its true peak is in the very first snapshot.

**Correction to earlier numbers:** a sampler's union holds only the sampled references that land in at least one bin, a fraction 1 − e^−λ of them, so its rate is (1 − e^−λ)/i, not 1/i. The first known-rate sampler numbers on this branch used 1/i, which overstated the sampled fraction 1.6×. They were 17% / 20% MAPE and are now 12% / 7%. `timeline.union_rate` recovers λ from the bin rows. Splitter unions were not affected.

## Improving the hybrid at `-i 25`

The steps were tested in the order below. All numbers are mean absolute error over the four kernels at `-i 25`, extrapolation / interpolation, in %.

| Step | MAPE | Error of peak | Error at peak |
|---|---|---|---|
| Before: correction trained on the splitter union with the nearest rate (1/37 for a 1/40 sampler) | 22.7 / 11.8 | 20.8 / 23.7 | 34.9 / 20.1 |
| 2. Correction trained at exactly the sampler's rate (splitter `-intervals 95,253,791,1582,3164,7910`) | 24.7 / 11.9 | 25.2 / 23.6 | 36.7 / 18.2 |
| 2 + 1. Estimate made non-decreasing between frees (isotonic per segment) | 23.8 / 10.1 | 19.8 / 15.7 | 37.9 / 19.0 |
| 2 + 3. One reuse shape per run, fitted on the run's data-rich snapshots | 24.2 / 11.7 | 30.5 / 20.6 | 46.3 / 18.7 |
| 2 + 1 on fresh runs (baseline for step 4) | 23.8 / 10.0 | 22.9 / 13.8 | 43.4 / 24.6 |
| 2 + 1 + 4. Feature: share of new addresses among the last 5 snapshots' samples (new `Discovered` counter) | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 |

- **Step 2** matters when the old rate mismatch was large. At `-i 50` (1.6× off before), extrapolation MAPE drops from 61% to 28–36%. At `-i 25`, where the mismatch was 8%, it changes nothing.
- **Step 1** reliably lowers the error of the peak, and is best at dense rates. At `-i 3` it gives 3.8–5.3% / 3.9–4.2% MAPE and 2.5–7% error of peak.
- **Step 3** does not help: the per-snapshot shape is no worse once the correction is applied.
- **Step 4** helps interpolation (MAPE 10.0 → 8.7) and the error at peak, but not extrapolation MAPE.
- **What remains is a size bias.** At `-i 25` the largest size is underestimated by 13–47% (2mm −11% to −36%, gemm −10% to −15%, jacobi-2d −28% to −47%, atax −27% to +11%), mostly in the first half of the run. Interpolation is almost unbiased.
  - The correction is learned on the smaller sizes, and at the same rate and phase the largest size's sample statistics fall outside that range.
  - None of the four steps addresses that.
- **Size-invariant features do not fix it either.** The bins' σ and mean (bytes) were replaced by the relative spread σ/mean and the bin/union footprint ratio, with smoothing and the discovery feature on, at `-i 25`:

  | Feature set | MAPE | Error of peak | Error at peak |
  |---|---|---|---|
  | Original (σ and bin mean in bytes) | 24.3 / 8.7 | 21.0 / 12.2 | 38.3 / 24.0 |
  | Size-invariant (ratios only) | 24.0 / 15.4 | 21.9 / 16.5 | 31.1 / 24.1 |

  - Per kernel, extrapolation MAPE improves for jacobi-2d (37.5 → 27.7) and atax (24.7 → 21.0) but worsens for 2mm (18.5 → 24.3) and gemm (16.3 → 23.1).
  - Interpolation gets worse, and at `-i 50`..`-i 250` it is unstable (43–72% interpolation MAPE, against 13–17%).
  - So the byte features carry useful information inside the training range and are not what causes the bias. The original set stays the default; the invariant one is `HYBRID_FEATURE_SETS["invariant"]`.
  - The bias more likely sits in the sample statistics themselves. In 2mm and gemm each element is reused about N times, so a larger input means more reuse per address. At the same sampling rate the largest size then shows more repeats than any training size did, whatever units the features use.
- **Remaining options:** train the correction on the larger sizes only (the tiniest ones may mislead it), constrain it to extrapolate monotonically in the size-related features, or add a feature for reuse per address relative to the known rate (e.g. the expected samples per address implied by the known-rate fit).

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
6. **A learned correction makes known-rate usable at sparser rates.**
   - The known-rate estimator's error is systematic: it underestimates while addresses have not yet been reused, which is exactly when f2/f1 is low. A regression on the training sizes learns this.
   - With the dense sampler, the corrected estimate is within 2–11% MAPE on all eight held-out runs.
   - With `-i 25` (about 1/5 of the cost) it gets 8–31%, against 35–102% raw.
   - At `-i 50` it is unstable.
7. **The error of the peak and the error at the peak are different questions.** The estimators get the maximum memory requirement right well before they get it right at the moment it happens. Example: raw known-rate on 2mm MEDIUM is +1% on the peak value but −19% at the peak time, because the estimate only catches up when the program starts re-reading its arrays halfway through.

## miniVite

Re-measured with `OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive`, using the f1/f2-only build, so Chao1 but not iChao1 or known-rate.

- **Stable run lengths:** with pinned, passive threads the sampler and splitter runs now agree within 1% on references executed. Unpinned, OpenMP threads spin and runs differed by up to ±50%.
- **Footprint not representative:** it is ~175 MB at every size from 1024 to 8192 vertices, against 7–14 MB in the paper's runs. The likely cause is that the system OpenMPI (`mpi/openmpi-x86_64`) touches large buffers at start-up; the paper used spack's openmpi-5.0.5. Until miniVite is built against that MPI, these numbers mostly measure MPI start-up memory.
- **Results on this data:**
  - α model: 22–25% (extrapolation) and 14–21% (interpolation) from splitter bins; 25–33% from the sampler.
  - Chao1: 34–35% from splitter unions; +219–241% from the sampler (the sparse-union overshoot).

## Open questions / next steps

- **Cost vs density:** measure the hybrid between `-i 3` and `-i 25` (e.g. 5, 8, 12) to find where it falls off, and its overhead at each.
- **Generalisation:** the correction is trained per workload on its smaller sizes, like the paper's α model. It has not been tested across workloads, or on miniVite.
- **Reuse model:** a two-class mixture (touched-once plus reused) instead of one negative binomial might transfer between sizes and allow sparser sampling.
- **miniVite:** build against spack openmpi-5.0.5 and re-measure with all three estimators.
- **Forecasting** is unchanged from `main`: the peak is within about 10–20% for interpolation and unreliable for extrapolation.
