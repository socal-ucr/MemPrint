# Second blind test: NPB IS, CG and MG, predicted from source before any trace

## Programs

| Program | Source | Build | Classes | Command line |
|---|---|---|---|---|
| IS (integer sort) | GMAP/NPB-CPP `5bc1e2c`, `NPB-SER/IS` (`workloads/npb`) | serial, `-O3` (NPB's make.def) | S, W | `is.<class>` |
| CG (conjugate gradient) | same, `NPB-SER/CG` | same | S, W | `cg.<class>` |
| MG (multigrid) | same, `NPB-SER/MG` | same | S, W, A | `mg.<class>` |

None of the three, nor anything else from NPB, was used to develop or calibrate the interpreter, the
skeleton models or the runtime baseline. Each generates its input from its class (compile-time
parameters from `sys/setparams`); MG would read `mg.input` if present, and the interpreter assumes no
input files (`fopen` returns NULL), as in the traced runs.

## Frozen code

The interpreter is frozen at commit `2ee17f3`. Bringing the programs up in the interpreter, before
any trace of them existed, found generic gaps, fixed in `d2258d8` and `df6e536` and listed in their
messages: dynamic initialisation of globals (NPB allocates its arrays in global initialisers),
globals read through their address (`randlc(&tran, ...)`), literal-only expressions folded at `-O3`,
variable-length arrays, `fopen`, loops passing `&scalar` to calls run in order, and loops of a few
rounds run in order. `df6e536`, `6b5c214` and `2ee17f3` also fix C++ library models, found while
recomputing GAP for the baseline (unordered_map and vector iterators, compare-and-swap loops,
`vector(n, value)`, vector field writes in batches); NPB uses no STL and no atomics.

The NPB spectra were computed from a git worktree at `df6e536`; the later commits change only library
models NPB does not use, and recomputing at `2ee17f3` gives bit-identical spectra (checked for MG at
every class and for IS and CG at class S; `identity.txt`). The GAP spectra behind the runtime
baseline were computed from a worktree at `2ee17f3`.

Classes are chosen by the interpreter's run time at the freeze (S: IS 17 min, CG 48 min, MG 10 s; W: IS 4.0 h,
CG 5.5 h, MG 68 s; A: MG 7 min; W
and A scale with the sequential random-number chains of IS and CG). Class A of IS and CG is not
predicted.

## Prediction

- Spectra: `runs.program_spectra` (the `static cpp` command) on a unity build of the kernel and NPB's
  common files (`paper_generalization/blind2/predict.py`), bytes touched, `-O3` counting, heap starting
  with 59,328 free bytes in glibc's top chunk (the GAP value, not measured for NPB).
- Runtime baseline: NNLS on the 12 serial GAP workloads at all traced scales with their interpreter
  spectra recomputed at `2ee17f3` (`data/gap-bytes/data/static_auto`).
- Predicted: the footprint (`k = 1`) and alpha at the 13 splitter intervals for every class, in
  `npb_<kernel>_predicted.csv`, committed before tracing.

## Scoring (fixed now)

- Trace with `scripts/run.sh npb --bench <npb_is|npb_cg|npb_mg> --configs "<classes>" --mode splitter --footprint bytes --runs 1`
  for the classes above.
- Per class: footprint error (predicted against Pin's exact footprint) and alpha MAPE over all bins of
  the 13 intervals.
- With two or three classes per program there is no own-model reference (the first blind test's EXTRA
  and INTER splits need at least three training configs); the errors are reported as they are, next
  to the first blind test's.
- The predictions are not changed after the traces exist. Anything learnt from the traces goes into a
  separate, labelled post-hoc analysis.

## Known weaknesses at the freeze

- MG: coverage 0.9996 at class S (a few floating-point branches on unknown values, charged half to
  each side).
- iostream output and NPB's result printing (`c_print_results`) are interpreted but their library
  internals are not modelled.
