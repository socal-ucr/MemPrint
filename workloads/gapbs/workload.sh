# GAP Benchmark Suite graph kernels on generated graphs. Configs are scales:
# gap_<kernel> uses uniform random graphs (-u SCALE), gap_<kernel>_kron
# Kronecker graphs (-g SCALE); 2^SCALE vertices, average degree 16.
# Built single-threaded (SERIAL=1) with GAP's default -O3. gap_pr runs a
# fixed number of iterations (-i 20 -t 0); gap_prc is the same binary with
# its defaults (-i 20 -t 1e-4), so it stops when it converges. Every kernel
# runs one trial; bc from one source, sssp with delta 1.

UPSTREAM_URL=https://github.com/sbeamer/gapbs.git
UPSTREAM_COMMIT=b5e3e19

DEFAULT_CONFIGS=(10 11 12 13 14 15 16)

# Benchmarks are gap_<kernel>[_kron] so that traces/<bench>/ matches the workload name.
wl_benchmarks() {
    local k
    for k in bfs pr prc cc sssp tc bc; do echo "gap_$k gap_${k}_kron"; done
}

_kernel() { local k=${1#gap_}; k=${k%_kron}; [[ $k == prc ]] && k=pr; echo "$k"; }

wl_build() { [[ -x $(_kernel "$1") ]] || make SERIAL=1 "$(_kernel "$1")"; }

wl_binary() { echo "$SRC/$(_kernel "$1")"; }

wl_args() {
    local graph="-u $2"
    [[ $1 == *_kron ]] && graph="-g $2"
    case $1 in
        gap_pr|gap_pr_kron) echo "$graph -k 16 -n 1 -i 20 -t 0" ;;
        *) echo "$graph -k 16 -n 1" ;;
    esac
}
