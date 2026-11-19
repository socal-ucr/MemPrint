MemRSS cluster runs (scripts/memrss_cluster.sh sim at commit 3ad47a9, monitor runs in
~/memprint-data/2026-11-19-memrss-cluster/runs). Every policy of tab:mpr-cluster in one run:
exact, request, mpr-exact, mpr, memrss, mpr+memrss, memrss-w, mpr+memrss-w, memrss-peak, mpr+memrss-peak.
  grid_none.csv      loads 0.4 to 1.2, seeds 0-2, truths rss and strict
  capacity_none.csv  load 0.8, local memory 64 to 256 GB per node (TB = (32 x local + 1024) / 1024)
  grid_native.csv    native costs, loads 0.6, 0.8, 1.2
