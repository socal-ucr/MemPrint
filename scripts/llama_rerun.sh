#!/bin/bash
# llama_rerun.sh: the llama.cpp runs whose reservation files have lines longer than 511 bytes (the key-value cache's
# "runs" lines, 60 to 120 ranges), which libmpr before the fix truncated (a 512-byte line buffer: it kept 24 ranges).
# Rerun with libmpr-v9.so (resize/runtime/mpr.c with an 8,192-byte line buffer; SONAME libmpr.so, so that it also
# stands in for the pass build's linked build/libmpr.so when preloaded). The old runs move to runs-trunc/ (and
# 2026-10-27-rsstrim/<set>-trunc/). Each part keeps its original configurations, reservation files and settings; the
# libmpr builds of Oct 12-13 had no MAP_NORESERVE thread stacks, so those parts run with MPR_STACKS=0.
#   PARTS (default "ml mlpass stacks nprof inputs inputsv2 trim")
set -u
R=$HOME/memprint-data/2026-10-12-resize; T=$HOME/memprint-data/2026-10-27-rsstrim
EVAL=$HOME/MemPrint/scripts/resize_eval.sh
V9=$R/libmpr-v9.so; export LIBMPR=$V9 OMP_WAIT_POLICY=passive
PARTS=" ${PARTS:-ml mlpass stacks nprof inputs inputsv2 trim} "
want() { [[ $PARTS == *" $1 "* ]]; }
keep() { # move an old run directory to the -trunc tree (once), make the new one
    local d=$1 old=${2:-${1/\/runs\//\/runs-trunc\/}}
    if [[ -d $d && ! -d $old ]]; then mkdir -p "$(dirname "$old")"; mv "$d" "$old"; else rm -rf "$d"; fi
    mkdir -p "$d"
}
LC=$R/ml/llamacpp.sh; GEN=$R/ml/llamacpp_gen.sh

if want ml; then   # ml/eval.sh: base, self, heldout, selfnz (libmpr.so of Oct 12)
    for name in llamacpp-135m-n32 llamacpp-135m-n128 llamacpp-360m-n64; do
        O=$R/runs/ml/$name; keep $O; cp $R/runs-trunc/ml/$name/res-*.txt $R/runs-trunc/ml/$name/fit-*.log $O/
        tag=${name#llamacpp-}; spec="/ - . $LC ${tag%-n*} ${tag##*-n}"
        for rep in 1 2 3; do
            $EVAL $O base - "$spec" base-$rep
            MPR_STACKS=0 $EVAL $O mpr $O/res-self.txt "$spec" self-$rep
            MPR_STACKS=0 $EVAL $O mpr $O/res-heldout.txt "$spec" heldout-$rep
            MPR_STACKS=0 $EVAL $O mprnz $O/res-self.txt "$spec" selfnz-$rep
        done
        echo "ml $name" >> $R/llama_rerun.done
    done
fi

if want mlpass; then   # ml/eval_pass.sh: the clang build with the pass, libmpr linked (preloaded v9 replaces it)
    W=$R/ml/llamacpp_pass_v9.sh
    printf '#!/bin/bash\n# as llamacpp_pass.sh, with libmpr-v9.so preloaded in place of the linked build/libmpr.so (same SONAME)\nD=$(cd "$(dirname "$0")" && pwd)\ncd "$D/llama.cpp/build-clang-pass/bin" && exec env LD_PRELOAD=%s ./llama-completion -m "$D/models/smollm2-$1-q8.gguf" -no-cnv -n "$2" --temp 0 \\\n    -s 1 -t 4 -p "Once upon a time" --simple-io < /dev/null\n' "$V9" > $W
    chmod +x $W
    for name in lcpass-135m-n32 lcpass-135m-n128 lcpass-360m-n64; do
        O=$R/runs/ml-pass/$name; keep $O; cp $R/runs-trunc/ml-pass/$name/res-*.txt $R/runs-trunc/ml-pass/$name/fit-*.log $O/
        tag=${name#lcpass-}; spec="/ - . $W ${tag%-n*} ${tag##*-n}"
        for rep in 1 2 3; do
            MPR_STACKS=0 $EVAL $O linked /dev/null "$spec" none-$rep
            MPR_STACKS=0 $EVAL $O linked $O/res-self.txt "$spec" self-$rep
            MPR_STACKS=0 $EVAL $O linked $O/res-heldout.txt "$spec" heldout-$rep
        done
        echo "mlpass $name" >> $R/llama_rerun.done
    done
fi

if want stacks; then   # stacks_eval.sh, llama.cpp part
    export OMP_NUM_THREADS=4
    for m in "135m 32" "360m 64"; do
        set -- $m; name=llamacpp-$1-n$2; O=$R/runs/stacks/$name; keep $O
        spec="/ - . $GEN $1 $2 0 1"
        for rep in 1 2 3; do
            $EVAL $O base - "$spec" base-$rep
            MPR_STACKS=0 $EVAL $O mpr $R/runs/inputs/res-llamacpp-$1.txt "$spec" nostacks-$rep
            MPR_STACKS=1 $EVAL $O mpr $R/runs/inputs/res-llamacpp-$1.txt "$spec" stacks-$rep
        done
        echo "stacks $name" >> $R/llama_rerun.done
    done
fi

if want nprof; then   # nprof_eval.sh, llama.cpp part: the profiles (prof/, res-native.txt, timing) are kept
    export OMP_NUM_THREADS=4
    for m in "135m 32" "360m 64"; do
        set -- $m; name=llamacpp-$1-n$2; O=$R/runs/nprof/$name; keep $O
        cp -r $R/runs-trunc/nprof/$name/{prof,res-native.txt,profile.time,profile.out,profile.err,fit.log} $O/
        spec="/ - . $GEN $1 $2 0 1"
        for rep in 1 2 3; do
            $EVAL $O base - "$spec" base-$rep
            MPR_STACKS=0 $EVAL $O mpr $R/runs/inputs/res-llamacpp-$1.txt "$spec" pin-$rep
            MPR_STACKS=0 $EVAL $O mpr $O/res-native.txt "$spec" native-$rep
        done
        echo "nprof $name" >> $R/llama_rerun.done
    done
fi

LCIN=("lc-135m-p2 135m 32 0 2" "lc-135m-p3 135m 32 0 3" "lc-135m-longprompt 135m 32 0 4" "lc-135m-n16 135m 16 0 1"
      "lc-135m-n256 135m 256 0 1" "lc-135m-n512 135m 512 0 1" "lc-135m-n1024 135m 1024 0 3" "lc-135m-c2048 135m 64 2048 1"
      "lc-135m-c4096 135m 64 4096 1" "lc-135m-sampled 135m 128 0 1 0.8 7" "lc-360m-n256 360m 256 0 2"
      "lc-360m-c4096 360m 128 4096 4")
if want inputs; then   # ml/inputs_eval.sh, llama.cpp part: base, mpr (pooled), model
    for x in "${LCIN[@]}"; do
        read -r name model args <<< "$x"; O=$R/runs/inputs/$name; keep $O
        for rep in 1 2 3; do
            $EVAL $O base - "/ - . $GEN $model $args" base-$rep
            MPR_STACKS=0 $EVAL $O mpr $R/runs/inputs/res-llamacpp.txt "/ - . $GEN $model $args" mpr-$rep
            MPR_STACKS=0 $EVAL $O mpr $R/runs/inputs/res-llamacpp-$model.txt "/ - . $GEN $model $args" model-$rep
        done
        echo "inputs $name" >> $R/llama_rerun.done
    done
fi

if want inputsv2; then   # ml/inputs_v2.sh and ml/inputs_one.sh, llama.cpp part: base, model, param (+ param1 for 135M)
    for x in "${LCIN[@]}"; do
        read -r name model args <<< "$x"; O=$R/runs/inputs-v2/$name; keep $O
        T_=$(awk -v n=$name '$1 == n {split($2, a, "="); print a[2]}' $R/runs/inputs-v2/params.txt)
        for rep in 1 2 3; do
            $EVAL $O base - "/ - . $GEN $model $args" base-$rep
            MPR_STACKS=0 $EVAL $O mpr $R/runs/inputs/res-llamacpp-$model.txt "/ - . $GEN $model $args" model-$rep
            MPR_STACKS=0 MPR_PARAMS=tokens=$T_ $EVAL $O mpr $R/runs/inputs-param/res-llamacpp-$model.txt "/ - . $GEN $model $args" param-$rep
        done
        if [[ $model == 135m ]]; then
            for rep in 1 2 3; do
                MPR_STACKS=0 MPR_PARAMS=tokens=$T_ $EVAL $O mpr $R/runs/inputs-param/res1-llamacpp-135m.txt "/ - . $GEN $model $args" param1-$rep
            done
        fi
        echo "inputsv2 $name" >> $R/llama_rerun.done
    done
fi

if want trim; then   # 2026-10-27-rsstrim: scripts/trim_eval.sh (libmpr-v8 there: stacks on by default, as v9)
    for w in llamacpp-135m-n32 llamacpp-135m-n128 llamacpp-360m-n64; do
        for set in runs runs_free runs_s0.02 runs_s0.3 runs_thread; do
            keep $T/$set/$w $T/$set-trunc/$w
        done
        OUT=$T/runs $HOME/MemPrint/scripts/trim_eval.sh $w > /dev/null 2>&1
        BUILDS="trim selftrim" MPR_TRIM=-1 MONITOR=1 OUT=$T/runs_free $HOME/MemPrint/scripts/trim_eval.sh $w > /dev/null 2>&1
        # the share sweep exported MPR_TRIM, so every build there trimmed (as it was run)
        for s in 0.02 0.3; do MPR_TRIM=$s OUT=$T/runs_s$s $HOME/MemPrint/scripts/trim_eval.sh $w > /dev/null 2>&1; done
        MONITOR=1 OUT=$T/runs_thread $HOME/MemPrint/scripts/trim_eval.sh $w > /dev/null 2>&1
        echo "trim $w" >> $R/llama_rerun.done
    done
fi
echo all-done >> $R/llama_rerun.done
