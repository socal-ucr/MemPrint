"""Array by array: references to each heap block under the hand skeleton and under the C++ interpreter.

python paper_generalization/arrays.py <scale> <kernel> [kron]   (appends to paper_generalization/data/gap_arrays.csv)
"""
import sys, collections
sys.path.insert(0, '/home/nmust004/MemPrint/analysis')
import numpy as np
from memprint.static import clangast as ca, interp_cpp, gap, skeleton, gap_kernels
from memprint.static import interp
scale = int(sys.argv[1]) if len(sys.argv) > 1 else 8
prog = sys.argv[2] if len(sys.argv) > 2 else 'pr'
# hand skeleton: per block refs, in allocation order
blocks = []
on, ot = skeleton.Process.new, skeleton.Process.touch
def new(self, nbytes, thread=0):
    b = on(self, nbytes, thread); b.refs = 0.0; blocks.append(b); return b
def touch(self, block, index, elem_size, refs=1.0):
    idx = np.arange(*index.indices(block.nbytes // elem_size)) if isinstance(index, slice) else np.asarray(index)
    block.refs = getattr(block, 'refs', 0.0) + float(np.sum(np.broadcast_to(np.asarray(refs, float), idx.shape)))
    return ot(self, block, index, elem_size, refs)
skeleton.Process.new, skeleton.Process.touch = new, touch
HAND = {'pr': gap.pagerank, 'bfs': gap.bfs, 'tc': gap_kernels.tc, 'bc': gap_kernels.bc, 'cc': gap_kernels.cc, 'sssp': gap_kernels.sssp}
kron = len(sys.argv) > 3
p = HAND[prog](scale, uniform=not kron) if kron else HAND[prog](scale)
hand = [(b.nbytes, b.refs) for b in blocks if b.space != 'stack']
CL = {'scale': scale, 'degree': 16, 'uniform': 1, 'symmetrize': 1, 'in_place': 0, 'filename': interp_cpp.Str(''),
      'num_trials': 1, 'max_iters': 20, 'tolerance': 0, 'logging_en': 0, 'do_analysis': 0, 'do_verify': 0,
      'start_vertex': -1, 'ParseArgs': 1, 'num_iters': 1, 'delta': 1}
CL['uniform'] = 0 if kron else 1
tu = ca.parse(f'/home/nmust004/MemPrint/workloads/src/gapbs/src/{prog}.cc', cplusplus=True)
made = []
r = interp.run(tu, footprint='bytes', heap_top=59328, make=lambda t, s: made.append(interp_cpp.CppInterpreter(t, s, {'__CL__': CL}, 3)) or made[-1])
it = made[0]
auto = []
for event, obj in it.heap_events:
    if event == 'new':
        refs = sum(float(a.sum()) for a in obj.by_size.values()) + obj.unresolved
        auto.append((obj.nbytes, refs, obj.name))
import csv, os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gap_arrays.csv')
new = not os.path.exists(out)
with open(out, 'a', newline='') as fh:
    wr = csv.writer(fh)
    if new: wr.writerow(['kernel', 'graph', 'scale', 'block', 'hand_bytes', 'hand_refs', 'auto_bytes', 'auto_refs'])
    for i in range(min(len(hand), len(auto))):
        wr.writerow([prog, 'kron' if kron else 'uniform', scale, i, hand[i][0], hand[i][1], auto[i][0], auto[i][1]])
print(f'{"#":>3} {"hand bytes":>10} {"hand refs":>12} | {"auto bytes":>10} {"auto refs":>12}  ratio')
for i in range(max(len(hand), len(auto))):
    h = hand[i] if i < len(hand) else (None, None)
    a = auto[i] if i < len(auto) else (None, None, '')
    ratio = (a[1] / h[1]) if h[1] and a[1] is not None else float('nan')
    print(f'{i:3d} {str(h[0]):>10} {h[1] if h[1] is None else round(h[1]):>12} | {str(a[0]):>10} {a[1] if a[1] is None else round(a[1]):>12}  {ratio:.3f}  {a[2] if len(a) > 2 else ""}')
print('totals hand', round(sum(h[1] for h in hand)), 'auto', round(sum(a[1] for a in auto)), 'interp total', round(it.total))
