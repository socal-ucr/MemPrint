# Template for a new workload. Copy this directory to workloads/<name>/ and
# fill in the fields below; scripts/setup.sh and scripts/run.sh need nothing
# else. Patches against the upstream source go in workloads/<name>/patches/
# (applied in name order with `git apply`).
#
# All functions run inside the checkout ($SRC = $WORKLOAD_SRC_DIR/<name>).

UPSTREAM_URL=https://example.com/project.git
UPSTREAM_COMMIT=0000000

# Input configurations run.sh iterates over when --configs is not given.
DEFAULT_CONFIGS=(small medium large)

# Optional: one-time preparation after checkout and patching.
# wl_setup() { ./configure; }

# Print the benchmark names, one per line.
wl_benchmarks() { echo project; }

# Build <bench> for <config>; a non-zero exit skips that config.
wl_build() { make; }

# Print the path of the executable for <bench> <config>.
wl_binary() { echo "$SRC/bin/$1"; }

# Optional: print the program arguments for <bench> <config> (default: none).
# wl_args() { echo "--size $2"; }

# Optional: print the trace name for <bench> <config> (default: <bench>-<config>).
# The analysis splits it at the last '-' into workload and config.
# wl_name() { echo "$1-$2"; }
