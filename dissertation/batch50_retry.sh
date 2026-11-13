#!/bin/bash
# batch50.sh's corpus traces that the OOM killer stopped (12 at a time exceeded 62 GB): their partial output removed,
# then rerun 3 at a time. Kept apart from batch50.sh, which was running when this step was added.
cd "$(dirname "$0")/.."; ROOT=$PWD
eval "$(sed -n '/^export PIN_ROOT/,/^export -f one/p' dissertation/batch50.sh)"
awk '$1=="c50" && $2=="split" && ($7!=0 || $8==0) {print $3, $4}' $O/done.txt | sort -u > $O/killed.txt
: > $O/jobs_retry.txt
while read -r w c; do
    rm -f $O/c50/traces/$w/Buffered_$w-${c}_1_* $O/c50/traces/$w/time.split.$c..
    corpus | grep "^$w $c " | sed 's/^/c50 split /' >> $O/jobs_retry.txt
done < $O/killed.txt
xargs -P 3 -L 1 bash -c 'one "$0" "$@"' < $O/jobs_retry.txt
