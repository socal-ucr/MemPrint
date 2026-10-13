"""Check the interpreter's per-site profiles (memprint.static.sites) on the Pin tool's -sites tests, whose sites have
known requested, touched and high-water bytes: python tests/static/test_sites.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from memprint.static import analyze, interp_cpp  # noqa: E402
from memprint.static import sites  # noqa: E402

KB, MB = 1024, 1024 * 1024
PIN = ROOT / "tests" / "pintool"
# (source, allocation line, first context line or None): (size, touched, high water) of each block there
EXPECTED = {
    ("sites.c", 8, None): [(1 * MB, 0, 0)],                            # dead
    ("sites.c", 13, None): [(1 * MB, 256 * KB, 256 * KB)],             # prefix
    ("sites.c", 21, None): [(512 * KB, 64 * KB, 512 * KB)],            # tail
    ("sites.c", 30, None): [(64 * KB, 4 * KB, 4 * KB)],                # calloc, 4 KB used
    ("sites.c", 50, None): [(16 * KB, 8 * KB, 8 * KB)] * 4,            # half
    ("sites_ctx.c", 12, 19): [(1 * MB, 1 * MB, 1 * MB)],               # my_alloc from hot()
    ("sites_ctx.c", 12, 26): [(1 * MB, 0, 0)],                         # my_alloc from cold()
    ("sites_vec.cpp", 9, None): [(1 * MB, 800000, 800000)],            # a vector of 100000 longs, capacity 2^17
}

ok = True
for src in sorted({s for s, _, _ in EXPECTED}):
    path = PIN / src
    if path.suffix == ".c":
        _, result, _, _ = analyze(str(path), footprint="bytes")
    else:
        _, result, _, _ = interp_cpp.analyze(str(path))
    b = sites.blocks(result.interp)
    for (s, line, ctx), want in EXPECTED.items():
        if s != src:
            continue
        g = b[b.site == f"src:{src}:{line}"]
        if ctx is not None:
            g = g[g.context.str.startswith(f"src:{src}:{ctx}")]
        got = sorted(zip(g["size"], g["touched"], g["high_water"]))
        good = got == sorted(want)
        print(f"{src}:{line}{'@' + str(ctx) if ctx else ''}: got {got} expected {sorted(want)} {'ok' if good else 'FAIL'}")
        ok &= good
sys.exit(0 if ok else 1)
