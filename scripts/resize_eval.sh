#!/bin/bash
# One program under one allocation configuration, for the resize study:
#   scripts/resize_eval.sh <out dir> <config> <reservations|-> "<cwd> <stdin|-> <grep pattern> <cmd...>"
# config: base (glibc), mpr (libmpr with the reservations), mprall (libmpr serving every large block),
#         mprscaled (reservations x MPR_SCALE, default 0.1: deliberate under-prediction).
# Writes <out dir>/<config>.csv (rss_sampler --commit), .out (the program's output), .report (libmpr's
# per-site telemetry) and appends "config peak_rss peak_commit seconds result_hash" to <out dir>/summary.txt.
set -u
out=$1 config=$2 res=$3
read -r cwd input pattern cmd <<< "$4"
HERE=$(cd "$(dirname "$0")" && pwd)
LIBMPR=${LIBMPR:-$HOME/memprint-data/2026-10-12-resize/libmpr.so}
mkdir -p "$out"
[[ $input == - ]] && input=/dev/null
read -r -a args <<< "$cmd"
for k in "${!args[@]}"; do [[ ${args[k]} == EMPTY ]] && args[k]=""; done
env=()
case $config in
  base) ;;
  mpr) env=(LD_PRELOAD="$LIBMPR" MPR_RESERVATIONS="$res" MPR_REPORT="$out/$config.report") ;;
  mprall) env=(LD_PRELOAD="$LIBMPR" MPR_RESERVATIONS="$res" MPR_ALL=1 MPR_REPORT="$out/$config.report") ;;
  mprscaled) env=(LD_PRELOAD="$LIBMPR" MPR_RESERVATIONS="$res" MPR_SCALE="${MPR_SCALE:-0.1}" MPR_REPORT="$out/$config.report") ;;
  *) echo "unknown config $config" >&2; exit 2 ;;
esac
cd "$cwd" || exit 1
start=$(date +%s.%N)
# only the program gets libmpr (env inside the sampled command), not the sampler
python3 "$HERE/rss_sampler.py" -i 0.02 --commit -o "$out/$config.csv" -- env "${env[@]}" "${args[@]}" \
    < "$input" > "$out/$config.out" 2> "$out/$config.err"
status=$?
secs=$(echo "$(date +%s.%N) - $start" | bc)
result=$(grep -E "$pattern" "$out/$config.out" | md5sum | cut -c1-12)
peaks=$(awk -F, 'NR > 1 { if ($2 > r) r = $2; if ($5 > c) c = $5 } END { print r + 0, c + 0 }' "$out/$config.csv")
echo "$config $peaks $secs $result $status" >> "$out/summary.txt"
