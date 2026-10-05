# miniVite (distributed Louvain community detection) on generated random
# geometric graphs. The single benchmark is miniVite; configs are vertex counts.
# Needs an MPI C++ compiler (mpicxx), e.g. `module load mpi/openmpi-x86_64`.

UPSTREAM_URL=https://github.com/nmustakin/miniVite.git
UPSTREAM_COMMIT=e630f9c

DEFAULT_CONFIGS=(1024 2048 4096 8192 16384)

wl_benchmarks() { echo miniVite; }

wl_build() { make; }

wl_binary() { echo "$SRC/miniVite"; }

wl_args() { echo "-n $2"; }

wl_name() { echo "miniVite-$2"; }
