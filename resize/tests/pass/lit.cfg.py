import os
import lit.formats
config.name = "MemPrintResize"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".ll"]
config.test_source_root = os.path.dirname(__file__)
llvm = os.environ.get("LLVM_BIN", os.path.expanduser("~/llvm-project/build/bin"))
config.environment["PATH"] = llvm + os.pathsep + os.environ["PATH"]
config.substitutions.append(("%plugin", os.environ["MPR_PLUGIN"]))
