#!/usr/bin/env python3
"""Run a command and sample the resident memory of its whole process tree.

    scripts/rss_sampler.py [-i 0.1] [-o rss.csv] [--limit BYTES] [--commit] -- cmd args...

Every interval (default 100 ms) it sums Rss over the command and all its descendants from
/proc/<pid>/smaps_rollup and writes rows (seconds, rss_bytes, pss_bytes, anon_bytes). It is the "peak RSS /
smaps sampling" baseline and the time-resolved RSS truth of the scheduling study.

--commit adds a column commit_bytes: the memory the kernel charges to the commit limit (Committed_AS), the
sizes of the mappings /proc/<pid>/smaps flags "ac" (private writable, not MAP_NORESERVE). Reading smaps is
slower than smaps_rollup.

--code adds a column code_bytes: the resident bytes of the tree's executable mappings (the program's and its
libraries' code, from /proc/<pid>/smaps), memory that does not grow with the program's data.

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


def commit(pid):
    """Bytes of the tree's accountable mappings (VmFlags ac)."""
    total = 0
    for p in descendants(pid):
        size = 0
        try:
            with open(f"/proc/{p}/smaps") as f:
                for line in f:
                    if line.startswith("Size:"):
                        size = int(line.split()[1]) * 1024
                    elif line.startswith("VmFlags:") and " ac" in line:
                        total += size
        except OSError:
            pass
    return total


def code(pid):
    """Resident bytes of the tree's executable mappings."""
    total, exe = 0, False
    for p in descendants(pid):
        try:
            with open(f"/proc/{p}/smaps") as f:
                for line in f:
                    first = line.split(None, 2)
                    if "-" in first[0] and len(first) > 1 and len(first[1]) == 4:   # a mapping's header line
                        exe = "x" in first[1]
                    elif exe and first[0] == "Rss:":
                        total += int(first[1]) * 1024
        except OSError:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=0.1)
    ap.add_argument("-o", "--out", default="rss.csv")
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--code", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)
    killed = False
    with open(a.out, "w") as out:
        out.write("seconds,rss_bytes,pss_bytes,anon_bytes" + (",commit_bytes" if a.commit else "")
                  + (",code_bytes" if a.code else "") + "\n")
        while proc.poll() is None:
            rss, pss, anon = sample(proc.pid)
            extra = (f",{commit(proc.pid)}" if a.commit else "") + (f",{code(proc.pid)}" if a.code else "")
            out.write(f"{time.monotonic() - t0:.3f},{rss},{pss},{anon}{extra}\n")
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
