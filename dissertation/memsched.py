"""Run shell commands in parallel within a memory budget: each input line is "<GB> <command...>"; a command starts
when its GB plus the GB of the running ones fits the budget (or nothing else is running), largest first.

    python3 dissertation/memsched.py <budget GB> <max jobs> < jobs.txt
"""
import subprocess
import sys
import time

budget, cap = float(sys.argv[1]), int(sys.argv[2])
jobs = []
for line in sys.stdin:
    if line.strip():
        gb, cmd = line.split(None, 1)
        jobs.append((float(gb), cmd.strip()))
jobs.sort(key=lambda j: -j[0])
running = []
while jobs or running:
    running = [(gb, p) for gb, p in running if p.poll() is None]
    used = sum(gb for gb, _ in running)
    for j in list(jobs):
        if len(running) >= cap:
            break
        if not running or used + j[0] <= budget:
            running.append((j[0], subprocess.Popen(["bash", "-c", j[1]])))
            used += j[0]
            jobs.remove(j)
    time.sleep(2)
