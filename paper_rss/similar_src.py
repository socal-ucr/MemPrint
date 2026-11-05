"""The code of each corpus program, for the code-similarity study (similar.py, similar_embed.py). No dependencies.

The code is what the job's author wrote: the program's sources for the compiled programs, and the script for the
programs run by an interpreter (lua, perl, python, sqlite), whose executable is the system's. cc1 has no code of its
own here (its input is a C file that it compiles, not runs, and GCC's sources are not on this machine), so it is left
out. Each program's binary is the executable that the corpus runs.

The patterns below mark the code that allocates, frees or grows memory (similar.py counts them; mem_slice cuts the
code around them for the embeddings).
"""
import glob
import os
import re

H = os.path.expanduser("~")
POLY = f"{H}/memprint-data/2026-10-0x-memory_estimator/e2e/src/polybench"
W = f"{H}/memory_estimator/workloads"
SRC = f"{H}/MemPrint/workloads/src"
SCRIPTS = f"{H}/memprint-data/2026-10-11-sched/corpus/inputs"
GAP_H = [f"{W}/gapbs/src/{h}" for h in ("benchmark.h", "bitmap.h", "builder.h", "command_line.h", "generator.h",
                                       "graph.h", "platform_atomics.h", "pvector.h", "reader.h", "sliding_queue.h",
                                       "timer.h", "util.h")]
NPB_COMMON = [f"{SRC}/npb/NPB-SER/common/{f}" for f in ("c_print_results.cpp", "c_randdp.cpp", "c_timers.cpp",
                                                        "npb-CPP.hpp", "wtime.cpp", "wtime.hpp")]


def _g(*pats):
    return sorted(f for p in pats for f in glob.glob(p, recursive=True))


def _poly(path, k):
    return [f"{POLY}/{path}/{k}/{k}.c", f"{POLY}/{path}/{k}/{k}.h"]


# program -> (language, source files, executable)
PROGRAMS = {
    "2mm": ("c", _poly("linear-algebra/kernels", "2mm"), f"{POLY}/linear-algebra/kernels/2mm/2mm-LARGE"),
    "atax": ("c", _poly("linear-algebra/kernels", "atax"), f"{POLY}/linear-algebra/kernels/atax/atax-LARGE"),
    "gemm": ("c", _poly("linear-algebra/blas", "gemm"), f"{POLY}/linear-algebra/blas/gemm/gemm-LARGE"),
    "jacobi-2d": ("c", _poly("stencils", "jacobi-2d"), f"{POLY}/stencils/jacobi-2d/jacobi-2d-LARGE"),
    **{f"gap-{k}": ("cpp", [f"{W}/gapbs/src/{k}.cc"] + GAP_H, f"{W}/gapbs/{k}") for k in ("bfs", "cc", "pr", "sssp")},
    "darknet": ("c", _g(f"{W}/darknet/src/*.c", f"{W}/darknet/src/*.h", f"{W}/darknet/include/*.h"),
                f"{W}/darknet/darknet"),
    "hpccg": ("cpp", _g(f"{SRC}/hpccg/*.cpp", f"{SRC}/hpccg/*.hpp"), f"{SRC}/hpccg/test_HPCCG"),
    "kripke": ("cpp", _g(f"{W}/Kripke/src/**/*.cpp", f"{W}/Kripke/src/**/*.h"), f"{W}/Kripke/build/kripke.exe"),
    "lulesh": ("cpp", _g(f"{SRC}/lulesh/*.cc", f"{SRC}/lulesh/*.h"), f"{SRC}/lulesh/lulesh2.0"),
    "minife": ("cpp", _g(f"{W}/MiniFE/source/*.cpp", f"{W}/MiniFE/source/*.hpp", f"{W}/MiniFE/source/*.c",
                         f"{W}/MiniFE/source/*.h"), f"{W}/MiniFE/source/miniFE.x"),
    "miniVite": ("cpp", _g(f"{H}/memprint-data/2026-10-0x-memory_estimator/mv-spack/minivite/*.hpp",
                           f"{H}/memprint-data/2026-10-0x-memory_estimator/mv-spack/minivite/*.cpp"),
                 f"{H}/memprint-data/2026-10-0x-memory_estimator/mv-spack/minivite/miniVite"),
    **{f"npb-{k.lower()}": ("cpp", [f"{SRC}/npb/NPB-SER/{k}/{k.lower()}.cpp", f"{SRC}/npb/NPB-SER/{k}/npbparams.hpp"]
                            + NPB_COMMON, f"{SRC}/npb/NPB-SER/bin/{k.lower()}.S") for k in ("CG", "IS", "MG")},
    "quicksilver": ("cpp", _g(f"{W}/Quicksilver/src/*.cc", f"{W}/Quicksilver/src/*.hh"), f"{W}/Quicksilver/src/qs"),
    "lua": ("lua", [f"{SCRIPTS}/lua_tab.lua"], "/usr/bin/lua"),
    "perl": ("perl", [f"{SCRIPTS}/pl_hash.pl"], "/usr/bin/perl"),
    "python": ("python", [f"{SCRIPTS}/py_json.py"], "/usr/bin/python3"),
    "sqlite": ("sql", [f"{SCRIPTS}/sql_1.sql"], "/usr/bin/sqlite3"),
}


def text(prog):
    """The program's code, its files in order, each headed by its name."""
    out = []
    for f in PROGRAMS[prog][1]:
        out.append(f"// file: {os.path.basename(f)}\n" + open(f, errors="replace").read())
    return "\n".join(out)


C_ALLOC = r"\b(malloc|calloc|aligned_alloc|posix_memalign|memalign|valloc|_mm_malloc|xmalloc|polybench_alloc_data|POLYBENCH_ALLOC_\w+)\s*\(|\bnew\b"
C_FREE = r"\bfree\s*\(|\bdelete\b|\bPOLYBENCH_FREE_ARRAY\b"
PATTERNS = {
    "c": dict(alloc=C_ALLOC, free=C_FREE),
    "cpp": dict(alloc=C_ALLOC, free=C_FREE),
    "lua": dict(alloc=r"\{", free=r"=\s*nil\b|collectgarbage"),
    "perl": dict(alloc=r"\$\w+\{|\$\w+\[|\[\s*\]|\{\s*\}", free=r"\bdelete\b|\bundef\b"),
    "python": dict(alloc=r"\{|\[|\bdict\(|\blist\(|json\.loads", free=r"\bdel\b"),
    "sql": dict(alloc=r"\bINSERT\b|\bCREATE\b", free=r"\bDELETE\b|\bDROP\b"),
}
COMMON = dict(realloc=r"\brealloc\s*\(",
              grow=r"\.(push_back|emplace_back|resize|reserve|insert|emplace|append)\s*\(|\bpush\s*\(?\s*@|table\.insert",
              node=r"\b(std::)?(map|set|list|deque|multimap|multiset|unordered_map|unordered_set)\s*<|%\w+|\bdict\b",
              vector=r"\b(std::)?vector\s*<", smart=r"make_shared|make_unique|shared_ptr|unique_ptr",
              omp=r"#\s*pragma\s+omp", mpi=r"\bMPI_\w+")


def strip(code, lang):
    if lang in ("c", "cpp"):
        code = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n") or " ", code, flags=re.S)   # lines kept
        code = re.sub(r"//[^\n]*", " ", code)
        code = re.sub(r'"(\\.|[^"\\\n])*"', '""', code)
    elif lang in ("perl", "python"):
        code = re.sub(r"#[^\n]*", " ", code)
    elif lang in ("lua", "sql"):
        code = re.sub(r"--[^\n]*", " ", code)
    return code


CONTEXT = 10   # lines kept before and after each line that matches a memory pattern


def mem_slice(prog):
    """The program's memory code: every line that allocates, frees or grows memory (PATTERNS alloc and free, COMMON
    realloc, grow, node and vector, on the code with comments and strings removed), with CONTEXT lines on each side,
    overlapping ranges merged, in file order; a script whole."""
    lang, files, _ = PROGRAMS[prog]
    if lang not in ("c", "cpp"):
        return text(prog)
    pat = re.compile("|".join([PATTERNS[lang]["alloc"], PATTERNS[lang]["free"]]
                              + [COMMON[k] for k in ("realloc", "grow", "node", "vector")]))
    out = []
    for f in files:
        raw = open(f, errors="replace").read()
        clean = strip(raw, lang)
        lines, keep = raw.splitlines(), set()
        for i, l in enumerate(clean.splitlines()):
            if pat.search(l):
                keep.update(range(max(i - CONTEXT, 0), min(i + CONTEXT + 1, len(lines))))
        if keep:
            parts, prev = [], -2
            for i in sorted(keep):
                if i != prev + 1:
                    parts.append("// ...")
                parts.append(lines[i]); prev = i
            out.append(f"// file: {os.path.basename(f)}\n" + "\n".join(parts))
    return "\n".join(out)
