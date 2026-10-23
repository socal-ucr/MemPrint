"""Usage: python tests/sched/check_ksplus_reference.py PATH/TO/KSPlus   (git clone https://github.com/dos-group/KSPlus)

Our KS+ port (analysis/memprint/sched/ksplus.py) against the authors' code (dos-group/KSPlus): segmentation,
prediction (regressions + offsets + sanitising) and the retry strategy, on random memory series."""
import ast, sys, types, itertools, bisect
import numpy as np
from sklearn.linear_model import LinearRegression
repo = sys.argv[1]
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
from memprint.sched import ksplus

# the authors' functions, taken from their source without importing their modules (which pull in matplotlib
# and the other baselines)
src = open(f'{repo}/tsb_resource_allocation/predictor.py').read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Segment' or
        isinstance(n, ast.FunctionDef) and n.name == 'get_segments']
ns = {'np': np}
exec(compile(ast.Module(body=keep, type_ignores=[]), 'predictor', 'exec'), ns)
msrc = ast.parse(open(f'{repo}/tsb_resource_allocation/model.py').read())
mkeep = [n for n in msrc.body if isinstance(n, ast.ClassDef) and n.name == 'Model']
for c in mkeep:   # drop the plotting method
    c.body = [b for b in c.body if not (isinstance(b, ast.FunctionDef) and b.name == 'plot')]
mns = {'bisect': bisect, 'itertools': itertools}
exec(compile(ast.Module(body=mkeep, type_ignores=[]), 'model', 'exec'), mns)
their_seg, Model = ns['get_segments'], mns['Model']

rng = np.random.default_rng(0)
bad = 0
for trial in range(2000):
    n = int(rng.integers(5, 120))
    data = list(np.round(np.cumsum(rng.normal(0, 50, n)) + 1000, 1))
    k = int(rng.integers(2, 7))
    a = their_seg(data, k)
    b = ksplus.get_segments(data, k)
    if list(a[0]) != list(b[0]) or not np.allclose(a[1], b[1]):
        bad += 1
print('segmentation mismatches:', bad, 'of 2000')

# prediction: their Predictor.__init__ + get_model, re-implemented from their source lines, against ksplus.fit
bad = 0
for trial in range(500):
    tasks = []
    for _ in range(int(rng.integers(2, 8))):
        size = float(rng.uniform(1, 100))
        n = int(rng.integers(8, 80))
        tasks.append((size, list(np.cumsum(np.abs(rng.normal(size, 20, n)))), 1.0))
    target = float(rng.uniform(1, 150))
    segs = [their_seg(m, 4) for _, m, _ in tasks]
    segends = [list(itertools.accumulate(st)) for st, _, _ in segs]
    X = np.array([[s] for s, _, _ in tasks])
    e = [LinearRegression().fit(X, [se[i] for se in segends]).predict([[target]])[0] * 0.85 for i in range(4)]
    p = [LinearRegression().fit(X, [sg[1][i] for sg in segs]).predict([[target]])[0] * 1.1 for i in range(4)]
    m = Model(e, p)
    ours_e, ours_p = ksplus.fit(tasks, target)
    # theirs floors peaks at 100 (MB) inside Model; ours applies the floor later, on the job's scale
    ref_p = list(m.segpeaks)
    ours_floored = list(np.maximum.accumulate([max(x, 100.0) for x in ours_p]))
    if not (np.allclose(np.maximum(m.segends, 0), ours_e) and np.allclose(ref_p, ours_floored)):
        bad += 1
print('prediction mismatches:', bad, 'of 500')

# retry: their get_model_failat against ksplus.fail_at (theirs truncates scaled ends to int: compare on integers)
bad = 0
for trial in range(2000):
    ends = sorted(int(x) for x in rng.integers(1, 200, 4))
    peaks = sorted(float(x) for x in rng.uniform(100, 1000, 4))
    t = int(rng.integers(0, 220))
    r = Model(list(ends), list(peaks)).get_model_failat(t)
    oe, op = ksplus.fail_at(list(map(float, ends)), list(peaks), float(t))
    if [int(x) for x in oe] != list(r.segends) or not np.allclose(op, r.segpeaks):
        bad += 1
print('retry mismatches:', bad, 'of 2000')
