#!/bin/bash
# Full E1: MemPrint on the scheduling corpus's own programs (2026-10-13-sched-llm/corpus/list.txt), so that the
# memprint-peak oracle draws each job's error from its own program. 22 programs: every program with three inputs
# whose two smaller inputs train the model (darknet has one input, llama.cpp two usable and NPB MG two with a truth).
#   split    the splitter on the two smaller inputs (13 intervals x 20 bins, bytes; miniVite -exclude_alloc 1)
#            -> $E/traces/<prog>/  (50 bins does not fit 62 GB at MiniFE 60, LULESH 45 or HPCCG 48)
#   truth    the largest input, one bin (-intervals 1000 -bins 1, bytes): MemPrint's truth, the bytes touched
#            -> $E/truth/<prog>/
#   sample   the bootstrap sampler (s=100, r=10, i=k/10, the high-bit tool) at the largest input at the L2O model's
#            training-best interval (corpus_e1.py build writes $E/sample_list.txt), seeds 1 to 3 -> $E/m3/...
#   dissertation/corpus_e1.sh split|truth|sample     (under memsched.py; existing outputs are skipped)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIN_ROOT=/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux OMP_WAIT_POLICY=passive
export E=$HOME/memprint-data/2026-11-23-corpus-e1 TOOL=$HOME/memprint-data/2026-11-23-hibits/memprint_trace_hibits.so
LISTS="$HOME/memprint-data/2026-10-13-sched-llm/corpus/list.txt $HOME/memprint-data/2026-10-13-sched-llm/corpus/list_llm.txt"
export LISTS
PROGS="2mm gemm jacobi-2d atax miniVite gap-bfs gap-cc gap-pr gap-sssp lua python perl sqlite cc1 hpccg lulesh npb-is npb-cg minife quicksilver kripke llama2c"
mkdir -p $E
export ROOT

one() {   # one <job> <prog> <input> <gb ignored> [seed k]   (the list line supplies threads, cwd and command)
    local job=$1 p=$2 i=$3; shift 3; local seed=${1:-} k=${2:-}
    local line; line=$(cat $LISTS | awk -v p=$p -v i=$i '$1 == p && $2 == i')
    local _p _i threads cwd cmd; read -r _p _i threads cwd cmd <<< "$line"
    local args; read -r -a args <<< "$cmd"   # as footprint_corpus.sh: EMPTY is an empty argument, _SP_ a space
    for x in "${!args[@]}"; do [[ ${args[x]} == EMPTY ]] && args[x]=""; args[x]=${args[x]//_SP_/ }; done
    local ex=(); [[ $p == miniVite ]] && ex=(-exclude_alloc 1)
    local d knobs
    case $job in
        split)  d=$E/traces/$p; knobs=(-mode splitter -bins 20 -footprint bytes "${ex[@]}")
                ls $d/Buffered_$p-${i}_1_*.csv > /dev/null 2>&1 && return ;;
        truth)  d=$E/truth/$p; knobs=(-mode splitter -intervals 1000 -bins 1 -footprint bytes "${ex[@]}")
                ls $d/Buffered_$p-${i}_*.csv > /dev/null 2>&1 && return ;;
        sample) d=$E/m3/boot/$seed/$p/$i/$k; [[ -d $d ]] && return
                knobs=(-mode sampler -i $((k / 10)) -s 100 -r 10 -footprint bytes -seed $seed "${ex[@]}") ;;
    esac
    mkdir -p $d; local run=$d; [[ $cwd == / ]] || run=$cwd
    (cd $run && OMP_NUM_THREADS=$threads /usr/bin/time -f "%e %M" -o $d/time.$job.$i.$seed \
        $PIN_ROOT/pin -t $TOOL "${knobs[@]}" -outdir $d -name $p-$i -- "${args[@]}" < /dev/null > /dev/null 2> $d/err.$job.$i.$seed)
    echo "$job $p $i ${seed:--} ${k:--} $? $(cat $d/time.$job.$i.$seed)" >> $E/done.txt; }
export -f one

gb() {   # memory estimate (GB) for a job: the corpus truth run's peak x factor, at least 1
    awk -v p=$1 -v i=$2 -v f=$3 '$1 == p && $2 == i && $3 == "truth" && $4 == 0 {g = $6 / 1e6 * f} END {printf "%.1f", (g > 1 ? g : 1)}' \
        $HOME/memprint-data/2026-10-13-sched-llm/corpus/done.txt; }

jobs=$E/jobs_$1.txt; : > $jobs
for p in $PROGS; do
    mapfile -t ins < <(cat $LISTS | awk -v p=$p '$1 == p {print $2}')
    case $1 in
        split)  for i in "${ins[0]}" "${ins[1]}"; do echo "$(gb $p $i 17) one split $p $i" >> $jobs; done ;;
        truth)  echo "$(gb $p ${ins[2]} 1.5) one truth $p ${ins[2]}" >> $jobs ;;
        sample) awk -v p=$p '$1 == p' $E/sample_list.txt | while read -r _ i k; do
                    for seed in 1 2 3; do echo "$(gb $p $i 17) one sample $p $i $seed $k" >> $jobs; done; done ;;
    esac
done
echo "$(wc -l < $jobs) jobs"
python3 $ROOT/dissertation/memsched.py ${BUDGET:-48} 16 < $jobs
echo "all-done $1" >> $E/done.txt
