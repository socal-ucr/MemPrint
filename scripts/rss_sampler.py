#!/usr/bin/env python3
"""Run a command and sample the resident memory of its whole process tree.

    scripts/rss_sampler.py [-i 0.1] [-o rss.csv] [--limit BYTES] -- cmd args...

Every interval (default 100 ms) it sums Rss over the command and all its descendants from
/proc/<pid>/smaps_rollup and writes rows (seconds, rss_bytes, pss_bytes, anon_bytes). It is the "peak RSS /
smaps sampling" baseline and the time-resolved RSS truth of the scheduling study.

--limit makes it an OOM watchdog: the tree is SIGKILLed the first time its RSS exceeds BYTES and the exit
status is 137, as under a cgroup memory.max. (cgroup v1 memory limits are not delegated to users on this
machine, so the limit is enforced by polling: a spike shorter than the interval can pass unseen.)
"""
import argparse
import os
import signal
import subprocess
import sys
import time

FIELDS = {"Rss:": 0, "Pss:": 1, "Anonymous:": 2}


def descendants(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    todo.extend(int(c) for c in f.read().split())
        except OSError:
            pass
    return out


def sample(pid):
    tot = [0, 0, 0]
    for p in descendants(pid):
        try:
            with open(f"/proc/{p}/smaps_rollup") as f:
                for line in f:
                    k = line.split(None, 1)[0]
                    if k in FIELDS:
                        tot[FIELDS[k]] += int(line.split()[1]) * 1024
        except OSError:
            pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=0.1)
    ap.add_argument("-o", "--out", default="rss.csv")
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)
    killed = False
    with open(a.out, "w") as out:
        out.write("seconds,rss_bytes,pss_bytes,anon_bytes\n")
        while proc.poll() is None:
            rss, pss, anon = sample(proc.pid)
            out.write(f"{time.monotonic() - t0:.3f},{rss},{pss},{anon}\n")
            if a.limit and rss > a.limit and not killed:
                os.killpg(proc.pid, signal.SIGKILL)
                killed = True
            time.sleep(a.interval)
    status = proc.wait()
    print(f"rss_sampler: {time.monotonic() - t0:.2f} s, exit {status}{' (killed over limit)' if killed else ''}",
          file=sys.stderr)
    sys.exit(137 if killed else (status if status >= 0 else 128 - status))


if __name__ == "__main__":
    main()
