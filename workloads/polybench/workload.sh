# PolyBench/C 4.2.1 with the extra MINI2/MINI3/SMALL2/SMALL3 dataset sizes.
# Benchmarks are kernel names (2mm, gemm, ...); configs are dataset sizes.

UPSTREAM_URL=https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git
UPSTREAM_COMMIT=3e87254

DEFAULT_CONFIGS=(MINI MINI2 MINI3 SMALL SMALL2 SMALL3 MEDIUM)

# Generate the per-kernel Makefiles. config.mk stays empty, so kernels are
# built by `cc` without optimisation flags, as for the traces in the paper.
wl_setup() {
    perl utilities/makefile-gen.pl . && : > config.mk
}

wl_benchmarks() {
    xargs -n1 basename -s .c < utilities/benchmark_list
}

_kernel_dir() {
    dirname "$(grep "/$1/$1.c\$" utilities/benchmark_list)"
}

wl_build() {
    make -C "$(_kernel_dir "$1")" SIZE="$2"
}

wl_binary() {
    echo "$SRC/$(_kernel_dir "$1")/$1-$2"
}
