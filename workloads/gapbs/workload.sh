# GAP Benchmark Suite graph kernels on generated graphs. Configs are scales:
# gap_<kernel> uses uniform random graphs (-u SCALE), gap_<kernel>_kron
# Kronecker graphs (-g SCALE); 2^SCALE vertices, average degree 16.
# Built single-threaded (SERIAL=1) with GAP's default -O3. gap_pr runs a
# fixed number of iterations (-i 20 -t 0); gap_prc is the same binary with
# its defaults (-i 20 -t 1e-4), so it stops when it converges. Every kernel
# runs one trial; bc from one source, sssp with delta 1. gap_<kernel>[_kron]_t4
# are OpenMP builds (<kernel>_omp); run them with OMP_NUM_THREADS=4.

UPSTREAM_URL=https://github.com/sbeamer/gapbs.git
UPSTREAM_COMMIT=b5e3e19

DEFAULT_CONFIGS=(10 11 12 13 14 15 16)

# Benchmarks are gap_<kernel>[_kron] so that traces/<bench>/ matches the workload name.
wl_benchmarks() {
    local k
    for k in bfs pr prc cc sssp tc bc; do echo "gap_$k gap_${k}_kron"; done
    for k in bfs pr; do echo "gap_${k}_t4 gap_${k}_kron_t4"; done
}

_kernel() { local k=${1#gap_}; k=${k%_t4}; k=${k%_kron}; [[ $k == prc ]] && k=pr; echo "$k"; }

_exe() { if [[ $1 == *_t4 ]]; then echo "$(_kernel "$1")_omp"; else _kernel "$1"; fi; }

wl_build() {
    local k; k=$(_kernel "$1")
    [[ -x $(_exe "$1") ]] && return
    if [[ $1 == *_t4 ]]; then
        ${CXX:-g++} -std=c++11 -O3 -Wall -fopenmp "src/$k.cc" -o "${k}_omp"
    else
        make SERIAL=1 "$k"
    fi
}

wl_binary() { echo "$SRC/$(_exe "$1")"; }

wl_args() {
    local graph="-u $2"
    [[ $1 == *_kron || $1 == *_kron_t4 ]] && graph="-g $2"
    case $1 in
        gap_pr|gap_pr_kron|gap_pr_t4|gap_pr_kron_t4) echo "$graph -k 16 -n 1 -i 20 -t 0" ;;
        *) echo "$graph -k 16 -n 1" ;;
    esac
}
