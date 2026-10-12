"""DMSched-Sim: job scheduling on disaggregated memory with a footprint oracle.

A time-slotted (1 slot = 1 minute) cluster of nodes with local DRAM and one shared memory pool. Jobs arrive,
an FCFS + EASY-backfill scheduler admits a job only if its whole reservation curve fits in a node's free
local memory plus the free pool from now to its projected end, and the only thing that changes between runs
is the footprint oracle that turns what is known about a job into that reservation curve. Any difference in
the results therefore comes from footprint knowledge alone.

- model.py: jobs (memory curves over progress p in [0, 1]) and the cluster;
- oracles.py: exact, exact peak, Massif-style curve, DHAT-style peak, and the measured estimators;
- sim.py: the scheduler, placement, remote-memory slowdown and OOM kills;
- synthetic.py: the synthetic job streams of the prototype.
"""
