# NAS Parallel Benchmarks 3.4 in C++ (GMAP NPB-CPP, serial version NPB-SER), built with its make.def
# (g++ -O3 -mcmodel=medium). Benchmarks npb_is (integer sort), npb_cg (conjugate gradient) and npb_mg
# (multigrid); configs are NPB problem classes (compile-time sizes). The second blind test.

UPSTREAM_URL=https://github.com/GMAP/NPB-CPP.git
UPSTREAM_COMMIT=5bc1e2c

DEFAULT_CONFIGS=(S W A)

wl_benchmarks() { echo npb_is npb_cg npb_mg; }

wl_build() {
    local k=${1#npb_}
    [[ -x NPB-SER/bin/$k.$2 ]] && return
    make -C NPB-SER "$k" CLASS="$2"
}

wl_binary() { echo "$SRC/NPB-SER/bin/${1#npb_}.$2"; }
