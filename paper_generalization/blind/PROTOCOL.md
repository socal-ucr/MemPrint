# Blind test of the C++ interpreter on programs never used in development

## Programs

| Program | Source | Build | Configs | Command line |
|---|---|---|---|---|
| HPCCG | Mantevo/HPCCG `80dd2f1` (`workloads/hpccg`) | serial, `-O3` (its Makefile, no MPI, no OpenMP) | grid edge n = 10 12 14 16 20 24 28 | `test_HPCCG n n n` (CG runs its fixed 150 iterations, tolerance 0) |
| LULESH 2.0 | LLNL/LULESH `3e01c40` (`workloads/lulesh`) | serial, `-O3`, `USE_MPI=0` | mesh edge s = 5 8 10 12 15 18 20 | `lulesh2.0 -q -s s -i 20` |

Neither program, nor any part of either, was used to develop or calibrate the interpreter or the
skeleton models. Both generate their input from their command line.

## Frozen code

The interpreter is frozen at commit `20a8d1e`. While bringing the two programs up in the interpreter,
before any trace of them existed, generic gaps were found and fixed in that commit (listed in its
message). After the freeze, one fix was made to the `static cpp` command's file handling (unity-build
include paths were relative, so no program parsed); it does not change what the interpreter computes.

## Prediction

- Spectra: `analysis/memprint/static/runs.program_spectra` (the `static cpp` command), bytes touched,
  `-O3` counting, with the heap starting with 59,328 free bytes in glibc's top chunk (the value measured
  for GAP, another libstdc++ program; not measured for these programs).
- Runtime baseline: fitted by NNLS on the 12 serial GAP workloads (uniform and Kronecker graphs; pr,
  bfs, cc, bc, tc, sssp) at all their traced scales, using their interpreter spectra (`data/gap-bytes`).
  Same compiler, libstdc++ and glibc; the programs' own I/O (iostream output, LULESH's and HPCCG's
  report files) is not modelled.
- Predicted: the footprint (`k = 1`) and alpha at the 13 splitter intervals, for every config.
  Written to `hpccg_predicted.csv` and `lulesh_predicted.csv` in this directory, and committed before
  tracing.

## Scoring (fixed now)

- Trace with `scripts/run.sh <hpccg|lulesh> --mode splitter --footprint bytes --runs 1`.
- Footprint error per config, and alpha MAPE over all bins of the 13 intervals per config.
- Splits as everywhere else: EXTRA (largest config held out) and INTER (middle config held out). The
  reference is the program's own MemPrint model trained on its other configs.
- The predictions are not changed after the traces exist. Anything learnt from the traces goes into a
  separate, labelled post-hoc analysis.

## Known weaknesses at the freeze

- LULESH: coverage 0.954. Its element loops copy node coordinates into local arrays through a nested
  loop; per-iteration values of those arrays are not kept across the nested batch, so some
  floating-point decisions (Courant and hydro time-step limits, small-value cut-offs) are unknown and
  charged half to each branch. Addresses come from the integer connectivity and are not affected.
- Neither program's iostream / file output is modelled.
