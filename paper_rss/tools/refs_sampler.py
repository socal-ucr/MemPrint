#!/usr/bin/env python3
"""Option 4: run a command; every -i seconds sum Rss, Anonymous and Referenced over its process tree
(smaps_rollup); every -w seconds, after sampling, clear the tree's referenced bits (clear_refs 1), so Referenced
is the pages touched since the window began.  -> seconds,rss_bytes,anon_bytes,ref_bytes,window"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts"))
from rss_sampler import descendants

a = argparse.ArgumentParser()
a.add_argument("-i", type=float, default=0.005); a.add_argument("-w", type=float, required=True)
a.add_argument("-o", required=True); a.add_argument("cmd", nargs=argparse.REMAINDER)
o = a.parse_args()
cmd = o.cmd[1:] if o.cmd[:1] == ["--"] else o.cmd
F = {"Rss:": 0, "Anonymous:": 1, "Referenced:": 2}
t0 = time.monotonic(); p = subprocess.Popen(cmd); win, nxt = 0, t0 + o.w
with open(o.o, "w") as out:
    out.write("seconds,rss_bytes,anon_bytes,ref_bytes,window\n")
    while p.poll() is None:
        tot = [0, 0, 0]; pids = descendants(p.pid)
        for q in pids:
            try:
                with open(f"/proc/{q}/smaps_rollup") as f:
                    for line in f:
                        k = line.split(None, 1)[0]
                        if k in F:
                            tot[F[k]] += int(line.split()[1]) * 1024
            except OSError:
                pass
        t = time.monotonic()
        out.write("%.4f,%d,%d,%d,%d\n" % (t - t0, *tot, win))
        if t >= nxt:
            for q in pids:
                try:
                    with open(f"/proc/{q}/clear_refs", "w") as f:
                        f.write("1")
                except OSError:
                    pass
            win += 1; nxt = t + o.w
        time.sleep(o.i)
sys.exit(p.returncode)
