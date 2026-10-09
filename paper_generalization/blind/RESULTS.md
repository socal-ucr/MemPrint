# Blind test results

Predictions committed at `2757a35`; traces taken afterwards (`data/blind`), scored by `score.py`
without changing the predictions.

| Program | footprint error | alpha MAPE per config | EXTRA alpha (own model) | INTER alpha (own model) |
|---|---|---|---|---|
| HPCCG, n = 10 ... 28 | +0.09 to +2.11% | 0.53 to 4.36% | 0.53% (12.24%) | 1.66% (11.05%) |
| LULESH, s = 5 ... 20 | -4.15 to -0.41% | 4.7 to 5.2% (s = 8 ... 12); 14.7% (s = 5); 28 to 43% (s = 15 ... 20) | 42.86% (7.87%) | 4.74% (17.03%) |

- HPCCG passes clearly: the footprint and alpha are predicted from source better than the program's own
  model trained on its traces.
- LULESH: the footprint is right everywhere, alpha at small and middle meshes, but alpha at the largest
  meshes is off by 28-43%. There the predicted spectrum has too few heavily referenced bytes
  (alpha at k = 10^5: 758 predicted, 467 measured at s = 20).

## Post hoc (after the traces): partial values

Looking for the cause, one interpreter defect was found: a variable assigned in the branches of an
if/else that different batch points take (LULESH's `rep`, the number of times a region's equation of
state is evaluated: 1, 2 or 20) stayed unknown, so those loops were charged one trip. With partial
values (`interp.Partial`, GAP spectra unchanged) `posthoc.py` gives alpha 8.3-24.9% at s = 10 ... 20
but 33.6% at s = 8: better at the largest meshes, worse at the middle ones. The blind numbers above are
the result of record; the remaining LULESH error has more than this one cause and is not resolved.
