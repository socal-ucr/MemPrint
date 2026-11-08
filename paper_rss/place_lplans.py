#!/usr/bin/env python3
"""Placement lines for libmpr (MPR_PLACE, "key place K lo1 hi1 ...") from an mprshim chunks profile (run_refs.sh, run
   "chunks"): the relative plans of place_rplans.py, with libmpr's site keys in place of mprshim's stack hashes.
   A site's key is its allocation call as libmpr names it, "image+0xaddress" (image: the base name dladdr gives, the
   address: the return address in the image file:
   the offset from the load base, plus the first segment's address for a non-PIE executable), with the DEPTH
   (default 2) frames above it as a context, "key@frame1@frame2". Frames in libstdc++ at the bottom of the stack
   (operator new) are skipped: libmpr sees new's caller. mprshim stacks that share a key are pooled.
   Each profiled 2 MiB chunk covers [c C / len, (c + 1) C / len) of its block; its memory-time (rss summed over the
   full windows) and referenced bytes are spread over the NB equal bins of the block it overlaps, by overlap.
   Units (key, bin) are ranked by referenced share: cold (least first), hot (most first), rand (seed 0); a plan
   takes units until their memory-time reaches f x the memory-time of all anonymous memory in the profile.
   python3 place_lplans.py PROFILE_OUT DEST -> DEST/<prog>/<policy><pct>.txt, DEST/plans.csv"""
import sys, os, struct, shutil, pandas as pd, numpy as np
PO, DEST = sys.argv[1], sys.argv[2]
NB, C = 16, 2097152
DEPTH = int(os.environ.get("DEPTH", 2))
FRACS = [0.25, 0.5]


def exe_base(path):
    """the first PT_LOAD address of a non-PIE executable (ET_EXEC), else 0"""
    try:
        with open(path, "rb") as f:
            h = f.read(64)
            if h[:4] != b"\x7fELF" or struct.unpack_from("<H", h, 16)[0] != 2: return 0
            phoff, = struct.unpack_from("<Q", h, 32); phentsize, phnum = struct.unpack_from("<HH", h, 54)
            f.seek(phoff); ph = f.read(phentsize * phnum)
            return min(struct.unpack_from("<Q", ph, i * phentsize + 16)[0] for i in range(phnum)
                       if struct.unpack_from("<I", ph, i * phentsize)[0] == 1)
    except OSError:
        return 0


def frame_key(obj, off, cwd):
    if "/" not in obj or obj.startswith("./"):   # the main program: libmpr names it by /proc/self/exe
        p = os.path.join(cwd, obj) if obj.startswith("./") else (shutil.which(obj) or obj)
    else:
        p = obj
    return f"{os.path.basename(obj)}+0x{int(off, 16) + exe_base(os.path.realpath(p)):x}"   # dladdr's name, as libmpr


def site_keys(s, cwd):
    keys = {}
    for site, g in s.sort_values(["site", "frame"]).groupby("site"):
        fr = [(o, off) for o, off in zip(g.object, g.offset)]
        while fr and os.path.basename(fr[0][0]).startswith("libstdc++"): fr = fr[1:]
        if not fr: continue
        keys[site] = "@".join(frame_key(o, off, cwd) for o, off in fr[:DEPTH + 1])
    return keys


lines = [l.split(None, 4) for l in open(f"{PO}/list.txt")]
progs = {l[0]: l[3] for l in lines}
rows = []
for prog, cwd in progs.items():
    d = f"{PO}/runs/{prog}/1"
    if not os.path.exists(f"{d}/chunks.csv.sites"):   # no block of 256 KiB or more (static arrays: PolyBench)
        print(prog, "no sites: no plan"); continue
    m, z = pd.read_csv(f"{d}/chunks.csv"), pd.read_csv(f"{d}/chunks.csv.zones")
    c, s = pd.read_csv(f"{d}/chunks.csv.chunks"), pd.read_csv(f"{d}/chunks.csv.sites", dtype={"sig": str, "offset": str})
    full = m.window[(m.window > 0) & (m.window < m.window.max())]
    z, c = z[z.window.isin(full)], c[c.window.isin(full)]
    total = z[z.zone.isin(["heap", "anon", "stack"]) | z.zone.str.startswith("site")].rss_bytes.sum()
    key = site_keys(s, cwd)
    k = c.groupby(["site", "chunk", "len"], as_index=False)[["rss_bytes", "ref_bytes"]].sum()
    k = k[k.site.isin(key)]
    r0 = (k.chunk * C / k.len).clip(upper=1).values; r1 = ((k.chunk + 1) * C / k.len).clip(upper=1).values
    acc = {}
    for i in range(len(k)):
        if r1[i] <= r0[i]: continue
        for b in range(int(r0[i] * NB), min(NB, int(np.ceil(r1[i] * NB)))):
            ov = (min(r1[i], (b + 1) / NB) - max(r0[i], b / NB)) / (r1[i] - r0[i])
            if ov <= 0: continue
            a = acc.setdefault((key[k.site.iat[i]], b), [0.0, 0.0])
            a[0] += ov * k.rss_bytes.iat[i]; a[1] += ov * k.ref_bytes.iat[i]
    u = pd.DataFrame([(kk, b, v[0], v[1]) for (kk, b), v in acc.items()], columns=["key", "bin", "mt", "ref"])
    u = u[u.mt > 0]; u["hot"] = u.ref / u.mt
    os.makedirs(f"{DEST}/{prog}", exist_ok=True)
    rng = np.random.default_rng(0)
    order = {"cold": u.sort_values(["hot", "mt"], ascending=[True, False]).index,
             "hot": u.sort_values(["hot", "mt"], ascending=[False, False]).index,
             "rand": u.index[rng.permutation(len(u))]}
    for f in FRACS:
        for pol, idx in order.items():
            uu = u.loc[idx]
            n = int(np.searchsorted(uu.mt.cumsum().values, f * total) + 1)
            sel = uu.iloc[:n]
            with open(f"{DEST}/{prog}/{pol}{int(f * 100)}.txt", "w") as out:
                for kk, g in sel.groupby("key"):
                    rs = []   # adjacent bins as one range
                    for b in sorted(g.bin):
                        if rs and rs[-1][1] == b: rs[-1][1] = b + 1
                        else: rs.append([b, b + 1])
                    out.write(f"{kk} place {len(rs)} " + " ".join(f"{a / NB:g} {e / NB:g}" for a, e in rs) + "\n")
            rows.append(dict(prog=prog, plan=f"{pol}{int(f * 100)}", planned_share=sel.mt.sum() / total, n_units=len(sel),
                             n_keys=sel.key.nunique(), mean_hot=np.average(sel.hot, weights=sel.mt)))
    rows.append(dict(prog=prog, plan="keys", n_units=len(set(key.values())), n_keys=len(key)))   # keys vs mprshim sites
P = pd.DataFrame(rows); P.to_csv(f"{DEST}/plans.csv", index=False)
print(P.round(3).to_string(index=False))
