# GAP Benchmark Suite graph kernels on generated uniform random graphs
# (-u SCALE: 2^SCALE vertices, average degree 16). Configs are scales.
# Built single-threaded (SERIAL=1) with GAP's default -O3. PageRank runs a
# fixed number of iterations (-i 20 -t 0, so convergence does not depend on
# the graph); every kernel runs one trial.

UPSTREAM_URL=https://github.com/sbeamer/gapbs.git
UPSTREAM_COMMIT=b5e3e19

DEFAULT_CONFIGS=(10 11 12 13 14 15 16)

# Benchmarks are gap_<kernel> so that traces/<bench>/ matches the workload name.
wl_benchmarks() { echo gap_bfs gap_pr; }

wl_build() { [[ -x ${1#gap_} ]] || make SERIAL=1 "${1#gap_}"; }

wl_binary() { echo "$SRC/${1#gap_}"; }

wl_args() {
    case $1 in
        gap_pr) echo "-u $2 -k 16 -n 1 -i 20 -t 0" ;;
        *)  echo "-u $2 -k 16 -n 1" ;;
    esac
}

