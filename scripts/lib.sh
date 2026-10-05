# Shared helpers for the MemPrint scripts.

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# Load workloads/<name>/workload.sh and point $SRC at its checkout.
load_workload() {
    local name=$1
    local descriptor="$MEMPRINT_ROOT/workloads/$name/workload.sh"
    [[ -f $descriptor ]] || die "unknown workload '$name' (no $descriptor)"
    WORKLOAD=$name
    WORKLOAD_DIR="$MEMPRINT_ROOT/workloads/$name"
    SRC="$WORKLOAD_SRC_DIR/$name"
    # Defaults that a descriptor may override
    wl_setup() { :; }
    wl_args() { :; }
    wl_name() { echo "$1-$2"; }
    source "$descriptor"
}

# Run a command and print its wall-clock time in seconds.
time_cmd() {
    local start end
    start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s.%N)
    awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f\n", e - s }'
}

# Average of the arguments.
mean() { printf '%s\n' "$@" | awk '{ s += $1 } END { printf "%.6f\n", s / NR }'; }

# Difference a - b.
minus() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f\n", a - b }'; }

# Peak heap reported by ms_print for a massif output file, as "<value> <unit>".
massif_peak() {
    ms_print "$1" | awk '
        /^\s*(B|KB|MB|GB)\s*$/ { u = $1 }
        /^[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*\^/ {
            match($0, /^[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*\^/, m)
            if (m[1] != "") v = m[1]
        }
        END { if (u != "" && v != "") print v, u }'
}

require_pin() {
    [[ -n $PIN_ROOT && -x $PIN_ROOT/pin ]] || die "set PIN_ROOT to the Pin kit (see README)"
    [[ -f $MEMPRINT_TOOL ]] || die "Pin tool not built: run 'make PIN_ROOT=\$PIN_ROOT' in pintool/"
}
