"""Check a -sites run of tests/pintool/sites.c (or sites_vec.cpp) against the sizes it was written with.

usage: check_sites.py OUTDIR BINARY {sites,sites_vec,sites_ctx}

Sites are named by the function their return address is in (addr2line -f on BINARY), so the
binary must be built with -g.
"""
import csv
import glob
import subprocess
import sys

KB, MB = 1024, 1024 * 1024
outdir, binary, which = sys.argv[1:4]
rows = list(csv.DictReader(open(glob.glob(f"{outdir}/*_sites.csv")[0])))
failed = False


def function(site):
    image, _, address = site.partition("+")
    if not binary.endswith(image):
        return image
    # -i: the whole inline chain (operator new is called from inlined allocator code); every function
    # name in it, innermost first, joined
    out = subprocess.run(["addr2line", "-f", "-i", "-C", "-e", binary, address], stdout=subprocess.PIPE,
                         universal_newlines=True).stdout.split("\n")
    return " < ".join(out[0::2][:-1] if out[-1] == "" else out[0::2])


by_function = {}
for r in rows:
    by_function.setdefault(function(r["site"]), []).append(r)


def check(name, fn, **expect):
    global failed
    got = [r for f, rs in by_function.items() for r in rs if fn in f.split(" < ")]
    if not got:
        print(f"    FAIL {name}: no site in {fn} (sites: {sorted(by_function)})")
        failed = True
        return
    r = got[0]
    bad = {k: (int(r[k]), v) for k, v in expect.items() if int(r[k]) != v}
    if bad:
        print(f"    FAIL {name}: " + ", ".join(f"{k} {g} != {v}" for k, (g, v) in bad.items()))
        failed = True
    else:
        print(f"    ok   {name}: " + ", ".join(f"{k} {v}" for k, v in expect.items()))


if which == "sites":
    check("never touched", "dead", count=1, requested=1 * MB, touched=0, high_water=0, never_touched=1)
    check("prefix touched", "prefix", requested=1 * MB, touched=256 * KB, high_water=256 * KB)
    check("tail touched", "tail", requested=512 * KB, touched=64 * KB, high_water=512 * KB)
    check("calloc zeroing a reused chunk", "zeroed", requested=64 * KB, alloc_touched=64 * KB)
    check("half of each block", "half", count=4, requested=64 * KB, touched=32 * KB, high_water=32 * KB)
    # realloc growth 1 KB .. 1 MB, each fully touched: grows in place at first, then moves
    grow = ([r for f, rs in by_function.items() for r in rs if "grow" in f.split(" < ")] or [{}])[0]
    if grow and int(grow["moves"]) >= 1 and int(grow["touched"]) == int(grow["requested"]):
        print(f"    ok   realloc growth: {grow['count']} blocks, {grow['moves']} moves, all touched")
    else:
        print(f"    FAIL realloc growth: {grow}")
        failed = True
elif which == "sites_ctx":
    blocks = list(csv.DictReader(open(glob.glob(f"{outdir}/*_blocks.csv")[0])))
    # the wrapper's blocks: one site, two contexts whose first frame is in hot() and in cold()
    wrapped = [b for b in blocks if "my_alloc" in function(b["site"]).split(" < ")]
    callers = {function(b["context"].split("|")[0]).split(" < ")[-1] if b["context"] else "": int(b["touched"])
               for b in wrapped}
    if len({b["site"] for b in wrapped}) == 1 and callers.get("hot") == 1 * MB and callers.get("cold") == 0:
        print(f"    ok   one wrapper site, two contexts: hot touched 1 MB, cold 0")
    else:
        print(f"    FAIL wrapper contexts: {[(b['site'], b['context'], b['touched']) for b in wrapped]} -> {callers}")
        failed = True
    cleared = [b for b in blocks if "cleared" in function(b["site"]).split(" < ")]
    if cleared and int(cleared[0]["touched"]) == 64 * KB and int(cleared[0]["high_water"]) == 64 * KB:
        print("    ok   memset to zero over untouched memory: touched 64 KB of 1 MB (high water 64 KB)")
    else:
        print(f"    FAIL cleared block: {cleared}")
        failed = True
else:
    vec = [r for f, rs in by_function.items() for r in rs if "grow_vector" in f or "_M_realloc_insert" in f]
    if vec and int(vec[0]["requested_max"]) == 1 * MB and int(vec[0]["count"]) >= 9:
        print(f"    ok   vector growth in the program: {vec[0]['count']} blocks, largest 1 MB "
              f"(high water {float(vec[0]['min_high_water_ratio']):.3f} of it)")
    else:
        print(f"    FAIL vector growth site not found in the program: {sorted(by_function)}")
        failed = True
sys.exit(failed)
