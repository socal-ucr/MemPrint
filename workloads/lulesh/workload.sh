# LULESH 2.0 (LLNL): Lagrangian shock hydrodynamics on an s x s x s hexahedral mesh that the program
# builds itself. Built serial (USE_MPI=0, no OpenMP) at -O3, as its Makefile's serial settings. Configs
# are the mesh edge s; every run does a fixed 20 cycles (-i 20) with output off (-q).

UPSTREAM_URL=https://github.com/LLNL/LULESH.git
UPSTREAM_COMMIT=3e01c40

DEFAULT_CONFIGS=(5 8 10 12 15 18 20)

wl_benchmarks() { echo lulesh; }

wl_build() {
    [[ -x lulesh2.0 ]] && return
    ${CXX:-g++} -DUSE_MPI=0 -O3 -I. -w lulesh.cc lulesh-comm.cc lulesh-viz.cc lulesh-util.cc lulesh-init.cc \
        -o lulesh2.0 -lm
}

wl_binary() { echo "$SRC/lulesh2.0"; }

wl_args() { echo "-q -s $2 -i 20"; }
