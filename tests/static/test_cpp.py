"""Check the C++ interpreter on tests/static/cpp_features.cc: python tests/static/test_cpp.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from memprint.static import clangast as ca  # noqa: E402
from memprint.static.interp_cpp import CppInterpreter  # noqa: E402
from memprint.static import interp  # noqa: E402

EXPECTED = [-1, 0, 5, 9, 10, 16, 5, 2, 20, 9, 14]

tu = ca.parse(Path(__file__).with_name("cpp_features.cc"), cplusplus=True)
made = []
interp.run(tu, footprint="bytes", make=lambda t, s: made.append(CppInterpreter(t, s)) or made[-1])
got = [int(v) if not isinstance(v, (interp.Ptr, type(interp.UNK))) else v for v in made[0].probes]
for i, (g, e) in enumerate(zip(got + [None] * len(EXPECTED), EXPECTED)):
    print(f"probe {i}: got {g!r:>8}  expected {e:>4}  {'ok' if g == e else 'FAIL'}")
sys.exit(0 if got == EXPECTED else 1)
