# The RSS model on paper_generalization's blind programs

The question: does the RSS footprint model, which has no per-program training, predict the footprint of the
programs that paper_generalization used for its two blind tests, at the same configurations and against the same
Pin truths?

## Programs and configurations (as paper_generalization/blind and blind2)

- HPCCG (`workloads/src/hpccg/test_HPCCG n n n`), n = 10 12 14 16 20 24 28
- LULESH 2.0 (`lulesh2.0 -q -s s -i 20`), s = 5 8 10 12 15 18 20
- NPB-CPP serial IS and CG at classes S and W, MG at S, W and A (A has no truth)

## Disclosure

The truths (Pin's footprint in bytes, `k = 1`, from paper_generalization) were traced and published before this
test, and their totals were printed in the session that wrote this protocol. The method below has no parameter that
is fitted to these programs, except the runtime constant of variant (b), which is fitted on the other programs only.

## Method (fixed before any run of these configurations)

- Run: each configuration 3 times under `paper_rss/tools/mprshim.c` in its pss mode, sampled every 1 ms
  (`MPR_INT=0.001`, as many configurations last a few ms) with one sample at exit; serial, in a scratch directory.
- Measured footprint of a configuration: the median over the 3 runs of the largest `anon_bytes - free_bytes` of the
  run (the data estimate without the runtime constant; these programs map no data files).
- Variant (a): the measured footprint alone.
- Variant (b): plus a runtime constant R, the median of (truth - measured) over the configurations of the other
  programs (leave one program out), for the bytes the runtime touches that anonymous RSS does not hold (read-only
  library data).
- Transfer (as paper_rss's transfer, earlier.transfer): EXTRA predicts the largest configuration with a truth from the
  others, INTER the middle one (HPCCG n = 16, LULESH s = 12) from the others; the peak from a linear regression of
  the measured footprint on the input size (n^3, s^3, NPB's sizes in sched.ksplus.INPUT_SIZE), never below the
  largest seen; with one other configuration, in proportion to the size. Plus R as in (b).
- Score: (prediction - truth) / truth in percent, per configuration and per split, next to paper_generalization's
  footprint errors (scores.csv) for the same configurations.
