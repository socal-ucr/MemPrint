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
parser.add_argument("--scale", action="store_true",
                    help="spatial runs: the footprint estimate is the selected footprint x SamplingInterval")
parser.add_argument("--chao", type=int, metavar="INTERVAL",
                    help="also check that Chao1 on the union row of this interval (Bin -2) "
                         "estimates the final unique addresses within --chao-error")
parser.add_argument("--chao-error", type=float, default=0.15)
args = parser.parse_args()

all_rows = list(csv.DictReader(open(args.timeline)))
rows = [r for r in all_rows if r["Bin"] == "-1"]
footprint = [int(r["MemUsageObs"]) * (int(r["SamplingInterval"]) if args.scale else 1) for r in rows]
peak, final, freed = max(footprint), footprint[-1], int(rows[-1]["FreedBytes"]) * (int(rows[-1]["SamplingInterval"]) if args.scale else 1)
final_max = args.slack if args.final_max is None else args.final_max

checks = [
    (f"peak {peak / MB:.3f} MB in [{args.peak - args.tolerance}, {args.peak + args.slack}]",
     (args.peak - args.tolerance) * MB <= peak <= (args.peak + args.slack) * MB),
    (f"freed {freed / MB:.3f} MB >= {args.freed}", freed >= args.freed * MB),
    (f"final {final / MB:.3f} MB <= {final_max}", final <= final_max * MB),
    (f"{len(rows)} snapshots", len(rows) >= 3),
]
if args.chao:
    last = rows[-1]["Time"]
    union = next(r for r in all_rows if r["Time"] == last and r["Bin"] == "-2" and r["SamplingInterval"] == str(args.chao))
    seen, f1, f2 = int(union["UniqueAddresses"]), int(union["Singletons"]), int(union["Doubletons"])
    chao = seen + f1 * (f1 - 1) / (2 * (f2 + 1))
    true = int(rows[-1]["UniqueAddresses"])
    checks.append((f"Chao1 {chao:.0f} vs {true} unique addresses ({seen / true:.1%} seen at 1-in-{args.chao})",
                   abs(chao - true) <= args.chao_error * true))
for text, ok in checks:
    print(f"    {'ok  ' if ok else 'FAIL'} {text}")
sys.exit(0 if all(ok for _, ok in checks) else 1)
