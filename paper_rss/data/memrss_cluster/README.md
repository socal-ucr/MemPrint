MemRSS cluster runs (scripts/memrss_cluster.sh sim at commit 3ad47a9, monitor runs in
~/memprint-data/2026-11-19-memrss-cluster/runs). Every policy of tab:mpr-cluster in one run:
exact, request, mpr-exact, mpr, memrss, mpr+memrss, memrss-w, mpr+memrss-w, memrss-peak, mpr+memrss-peak.
  grid_none.csv      loads 0.4 to 1.2, seeds 0-2, truths rss and strict
  capacity_none.csv  load 0.8, local memory 64 to 256 GB per node (TB = (32 x local + 1024) / 1024)
  grid_native.csv    native costs, loads 0.6, 0.8, 1.2
Where MemRSS's estimate is the target (commit after 801dea5, same policies, RESIZE_EST_RUNS as above):
  grid_reclaim.csv     truth "reclaim": a node that takes the allocator's free bytes back (each glibc or libmpr
                       run's RSS less the free bytes of the monitored run of the same index), loads 0.4 to 1.2
  capacity_reclaim.csv the same at load 0.8 over local memory 64 to 256 GB per node
  grid_inflate.csv     requests 1.3, 1.6 and 2 times the program's allocation (--inflate 0.3 0.6 1.0), truths rss and
                       reclaim, loads 0.4 to 1.2
