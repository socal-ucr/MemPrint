"""Check the C++ interpreter's library models on tests/static/umap.cc (unordered_map iterators) and
tests/static/cas.cc (compare_and_swap idioms) and
tests/static/globals.cc (globals through their address): python tests/static/test_library.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from memprint.static import clangast as ca  # noqa: E402
from memprint.static.interp_cpp import CppInterpreter  # noqa: E402
from memprint.static import interp  # noqa: E402

EXPECTED = {"umap.cc": [2, 41, 66, 4, 100, 111, 1, 1, 1, 3, 5], "cas.cc": [64, 1, 8],
            "globals.cc": [314159265, 1220703125, 314159265, 119209, 8388608, 794521, 869065, 647631, 15]}
ok = True
for name, expected in EXPECTED.items():
    tu = ca.parse(Path(__file__).with_name(name), cplusplus=True)
    made = []
    interp.run(tu, footprint="bytes", make=lambda t, s: made.append(CppInterpreter(t, s)) or made[-1])
    got = [v if isinstance(v, (interp.Ptr, type(interp.UNK))) else int(v) for v in made[0].probes]
    print(f"{name}: got {got} expected {expected} {'ok' if got == expected else 'FAIL'}")
    ok &= got == expected
sys.exit(0 if ok else 1)
