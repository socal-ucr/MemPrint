# GAP Benchmark Suite graph kernels on generated graphs. Configs are scales:
# gap_<kernel> uses uniform random graphs (-u SCALE), gap_<kernel>_kron
# Kronecker graphs (-g SCALE); 2^SCALE vertices, average degree 16.
# Built single-threaded (SERIAL=1) with GAP's default -O3. PageRank runs a
# fixed number of iterations (-i 20 -t 0, so convergence does not depend on
# the graph); every kernel runs one trial.

UPSTREAM_URL=https://github.com/sbeamer/gapbs.git
UPSTREAM_COMMIT=b5e3e19

DEFAULT_CONFIGS=(10 11 12 13 14 15 16)

# Benchmarks are gap_<kernel>[_kron] so that traces/<bench>/ matches the workload name.
wl_benchmarks() { echo gap_bfs gap_pr gap_bfs_kron gap_pr_kron; }

_kernel() { local k=${1#gap_}; echo "${k%_kron}"; }

wl_build() { [[ -x $(_kernel "$1") ]] || make SERIAL=1 "$(_kernel "$1")"; }

wl_binary() { echo "$SRC/$(_kernel "$1")"; }

wl_args() {
    local graph="-u $2"
    [[ $1 == *_kron ]] && graph="-g $2"
    case $(_kernel "$1") in
        pr) echo "$graph -k 16 -n 1 -i 20 -t 0" ;;
        *)  echo "$graph -k 16 -n 1" ;;
    esac
}
