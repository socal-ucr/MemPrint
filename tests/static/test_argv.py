"""Check command-line arguments in the interpreters (atoi / stol / strtol of argv[i]):
python tests/static/test_argv.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from memprint.static import clangast as ca, interp  # noqa: E402
from memprint.static.interp_cpp import CppInterpreter  # noqa: E402

EXPECTED = [4, 37, 1000, 16, 36]  # argc, atoi(argv[1]), std::stol(argv[2]), strtol(argv[3]) of "0x10", a[n-1]

tu = ca.parse(Path(__file__).with_name("argv.cc"), cplusplus=True)
made = []
interp.run(tu, footprint="bytes", argv=["argv", "37", "1000", "0x10"],
           make=lambda t, s: made.append(CppInterpreter(t, s)) or made[-1])
got = [int(v) if isinstance(v, int) else v for v in made[0].probes]
print("got", got, "expected", EXPECTED, "ok" if got == EXPECTED else "FAIL")
sys.exit(0 if got == EXPECTED else 1)
