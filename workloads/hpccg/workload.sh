# HPCCG (Mantevo): conjugate gradient on a 27-point stencil matrix that the program generates for an
# nx x ny x nz grid per process. Built serial (no MPI, no OpenMP) with its Makefile's -O3. Configs are
# the grid edge n (nx = ny = nz = n); CG runs its fixed 150 iterations (tolerance 0).

UPSTREAM_URL=https://github.com/Mantevo/HPCCG.git
UPSTREAM_COMMIT=80dd2f1

DEFAULT_CONFIGS=(10 12 14 16 20 24 28)

wl_benchmarks() { echo hpccg; }

wl_build() {
    [[ -x test_HPCCG ]] && return
    make USE_MPI= USE_OMP= CXX="${CXX:-g++}" LINKER="${CXX:-g++}"
}

wl_binary() { echo "$SRC/test_HPCCG"; }

wl_args() { echo "$2 $2 $2"; }
