#!/usr/bin/env python3
"""Which of a reservation file's pass ids exist in a build (its MemPrintResize site map).

    scripts/check_reservations.py RESERVATIONS SITEMAP

Pass ids (id:0x..., cap:0x...) are stable across rebuilds unless the code around a site changes
(they hash each inlining level's line relative to its function); this lists the ones a new build
no longer has, which libmpr would silently not apply. Address keys (image+0x...) are checked by
libmpr itself, against the build-id recorded with the reservations.
"""
import sys

res, sitemap = sys.argv[1:3]
ids = {line.split()[0].lower() for line in open(sitemap) if line.strip()}
keys = [line.split()[0] for line in open(res) if line.strip() and not line.startswith("#")]
mine = [k for k in keys if k.split("@")[0].startswith(("id:", "cap:"))]
missing = [k for k in mine if k.split("@")[0].split(":", 1)[1].lower() not in ids]
print(f"{len(mine) - len(missing)} of {len(mine)} pass sites found in {sitemap}; "
      f"{len(keys) - len(mine)} address keys (checked by build-id at run time)")
for k in missing:
    print("  missing", k)
sys.exit(1 if missing else 0)
