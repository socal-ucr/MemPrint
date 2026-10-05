"""Check a memprint_trace timeline against expected footprint bounds.

usage: check.py TIMELINE --peak MB --freed MB [--final-max MB] [--slack MB]

Peak live footprint must lie in [peak - tolerance, peak + slack]: snapshots
are taken every N references, so the true peak can fall between two of them,
and slack covers the C runtime's own memory. At least `freed` must have been
released (more is fine: the footprint counts the largest access at each start
address, so overlapping vector accesses, e.g. in memcpy, weigh more than their
bytes), and the footprint at exit must be below final-max (default: slack).
"""
import argparse
import csv
import sys

MB = 1024 * 1024

parser = argparse.ArgumentParser()
parser.add_argument("timeline")
parser.add_argument("--peak", type=float, required=True)
parser.add_argument("--freed", type=float, default=0)
parser.add_argument("--final-max", type=float)
parser.add_argument("--slack", type=float, default=0.5)
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()

rows = [r for r in csv.DictReader(open(args.timeline)) if r["Bin"] == "-1"]
footprint = [int(r["MemUsageObs"]) for r in rows]
peak, final, freed = max(footprint), footprint[-1], int(rows[-1]["FreedBytes"])
final_max = args.slack if args.final_max is None else args.final_max

checks = [
    (f"peak {peak / MB:.3f} MB in [{args.peak - args.tolerance}, {args.peak + args.slack}]",
     (args.peak - args.tolerance) * MB <= peak <= (args.peak + args.slack) * MB),
    (f"freed {freed / MB:.3f} MB >= {args.freed}", freed >= args.freed * MB),
    (f"final {final / MB:.3f} MB <= {final_max}", final <= final_max * MB),
    (f"{len(rows)} snapshots", len(rows) >= 3),
]
for text, ok in checks:
    print(f"    {'ok  ' if ok else 'FAIL'} {text}")
sys.exit(0 if all(ok for _, ok in checks) else 1)
