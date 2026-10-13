#!/bin/bash
# Footprint corpus for the scheduling study: one program run under one footprint method.
#
#   scripts/footprint_corpus.sh <method> "<prog> <input> <threads> <cwd> <cmd...>"
#
# Writes $CORPUS/<prog>/<input>/<method>/ (the method's raw output) and appends
# "<prog> <input> <method> <exit> <wall s> <max RSS KB>" to $CORPUS/done.txt (the cost of the method).
# EMPTY in the command stands for an empty argument, and _SP_ for a space inside one (a prompt). Methods:
#   native       the program alone, its RSS every 5 ms (rss.csv) and instruction count (perf.txt)
#   truth        MemPrint splitter, live bytes touched over time (-track_frees), the full Pin estimate
#   W1 | W5      MemPrint windowed, 1% / 5% of the run watched
#   P            MemPrint pages only (page residency x density, nothing watched)
#   truthpages   truth counting 4 KB pages touched (-footprint pages)
#   W5pages      W5 estimating resident pages (-footprint pages)
#   W5sec        W5 again (with a tool that writes the Seconds column)
#   massif       Massif, heap, time in ms
#   massifpages  Massif --pages-as-heap
#   massifi | massifpagesi   the same with Massif's default clock, instructions executed (--time-unit=i)
#   dhat         DHAT
#   heaptrack    heaptrack
#   gperf        gperftools heap profiler (tcmalloc), a dump every 1/50 of the native peak RSS
# The snapshot and window lengths come from the native run's instruction count, so run native first.
set -u
method=$1
read -r prog input threads cwd cmd <<< "$2"
CORPUS=${CORPUS:-$HOME/memprint-data/2026-10-11-sched/corpus}
PIN=${PIN:-/home/nmust004/memory_estimator/pin/pin-3.30-98830-g1d7b601b3-gcc-linux/pin}
TOOL=${TOOL:-$CORPUS/tool.so}
HERE=$(cd "$(dirname "$0")" && pwd)
GPERF=/home/nmust004/spack/opt/spack/linux-broadwell/gperftools-2.18-u4cqvcpknhrzftdrxadqtow2rchwko4l/lib
HEAPTRACK=${HEAPTRACK:-/home/nmust004/spack/opt/spack/linux-broadwell/heaptrack-1.3.0-fob5atwjkybije52useb6lqyrp43x4zp/bin/heaptrack}
export PATH=/home/nmust004/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/openmpi-5.0.5-hwdt6fs2qm5ybyrq4u73lbgdv6utoz6x/bin:$PATH
export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive

out=$CORPUS/$prog/$input/$method
rm -rf "$out"; mkdir -p "$out"
read -r -a args <<< "$cmd"
for k in "${!args[@]}"; do [[ ${args[k]} == EMPTY ]] && args[k]=""; args[k]=${args[k]//_SP_/ }; done
stdin=/dev/null
[[ $prog == darknet ]] && stdin=$HOME/memprint-data/2026-10-0x-memory_estimator/win/cand/images.txt

native=$CORPUS/$prog/$input/native
if [[ $method != native ]]; then
    instr=$(awk -F, '$3 ~ /instructions/ {print $1}' "$native/perf.txt")
    snap=$((instr / 1000 > 1000 ? instr / 1000 : 1000))   # about 500 snapshots at ~0.5 references per instruction
    period=$((instr / 40 > 100000 ? instr / 40 : 100000))  # about 20 windows
    peak_kb=$(awk -F, 'NR > 1 && $2 > m {m = $2} END {print int(m / 1024)}' "$native/rss.csv")
fi

cd "$cwd" || exit 1
TIMEFILE=$out/time.txt
timed() { /usr/bin/time -o "$TIMEFILE" -f "%e %M" "$@"; }
case $method in
  native)      perf stat -x, -e instructions:u -o "$out/perf.txt" "${args[@]}" < $stdin > /dev/null 2>&1
               timed python3 "$HERE/rss_sampler.py" -i 0.005 -o "$out/rss.csv" -- "${args[@]}" ;;
  truth)       timed "$PIN" -t "$TOOL" -mode splitter -intervals 1000 -bins 1 -track_frees 1 -snapshot $snap -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  truthpages)  timed "$PIN" -t "$TOOL" -mode splitter -intervals 1000 -bins 1 -track_frees 1 -snapshot $snap -footprint pages -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  W5pages)     timed "$PIN" -t "$TOOL" -mode spatial -i 100 -s 20 -snapshot $snap -window $((period / 20)) -period $period -window_alloc 0 -footprint pages -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  W5sec)       timed "$PIN" -t "$TOOL" -mode spatial -i 100 -s 20 -snapshot $snap -window $((period / 20)) -period $period -window_alloc 0 -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  W1)          timed "$PIN" -t "$TOOL" -mode spatial -i 100 -s 20 -snapshot $snap -window $((period / 100)) -period $period -window_alloc 0 -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  W5)          timed "$PIN" -t "$TOOL" -mode spatial -i 100 -s 20 -snapshot $snap -window $((period / 20)) -period $period -window_alloc 0 -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  P)           timed "$PIN" -t "$TOOL" -mode spatial -i 100 -s 20 -snapshot $snap -pages_only 1 -outdir "$out" -name "$prog-$input" -- "${args[@]}" ;;
  massif)      timed valgrind --tool=massif --time-unit=ms --max-snapshots=1000 --detailed-freq=1000000 \
                   --massif-out-file="$out/massif.out" "${args[@]}" ;;
  massifpages) timed valgrind --tool=massif --pages-as-heap=yes --time-unit=ms --max-snapshots=1000 \
                   --detailed-freq=1000000 --massif-out-file="$out/massif.out" "${args[@]}" ;;
  massifi)     timed valgrind --tool=massif --time-unit=i --max-snapshots=1000 --detailed-freq=1000000 \
                   --massif-out-file="$out/massif.out" "${args[@]}" ;;
  massifpagesi) timed valgrind --tool=massif --pages-as-heap=yes --time-unit=i --max-snapshots=1000 \
                   --detailed-freq=1000000 --massif-out-file="$out/massif.out" "${args[@]}" ;;
  dhat)        timed valgrind --tool=dhat --dhat-out-file="$out/dhat.json" "${args[@]}" ;;
  heaptrack)   timed "$HEAPTRACK" -o "$out/heaptrack" "${args[@]}" ;;
  gperf)       timed env LD_PRELOAD=$GPERF/libtcmalloc_and_profiler.so HEAPPROFILE="$out/heap" \
                   HEAP_PROFILE_INUSE_INTERVAL=$((peak_kb * 1024 / 50 > 65536 ? peak_kb * 1024 / 50 : 65536)) \
                   HEAP_PROFILE_ALLOCATION_INTERVAL=0 "${args[@]}" ;;
  *) echo "unknown method $method" >&2; exit 2 ;;
esac < $stdin > "$out/stdout.txt" 2> "$out/stderr.txt"
status=$?
echo "$prog $input $method $status $(cat "$TIMEFILE" 2>/dev/null | tail -1)" >> "$CORPUS/done.txt"
