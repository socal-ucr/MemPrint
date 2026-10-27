#!/bin/bash
# The alternatives to the model, each alone, on the 7 allocator-heavy programs at their largest inputs, 3 repetitions,
# one run at a time (paper_rss/compute.py: options). Writes $D/runs/<prog>/<rep>/{base,opt1,opt2a,opt2b,opt3,opt4}.csv
# and $D/done.txt. opt3 exists for python (pymalloc statistics) and lua (collectgarbage("count")) only.
HERE=$(cd "$(dirname "$0")" && pwd); S=$HERE/../scripts; D=${OUT:-$HOME/memprint-data/2026-10-26-rsscurve/opts}
gcc -O2 -shared -fPIC -o $D/mprshim.so $HERE/tools/mprshim.c -lpthread || exit 1
for rep in 1 2 3; do
grep -E "^(miniVite 32768|lua 1|python 1|perl 1|sqlite 1|cc1 large|quicksilver 100000) " $D/../list.txt | \
while read -r prog input threads cwd cmd; do
    export OMP_NUM_THREADS=$threads OMP_WAIT_POLICY=passive
    read -r -a args <<< "$cmd"
    wall=$(awk -v p=$prog -v i=$input '$1==p && $2==i && $3=="rssrep1" {print $5}' $D/../corpus/done.txt)
    o=$D/runs/$prog/$rep; mkdir -p $o; cd "$cwd"
    t() { m=$1; shift; /usr/bin/time -o $o/$m.time -f "%e %M" "$@" < /dev/null > /dev/null 2> $o/$m.err; echo "$prog $rep $m $? $(tail -1 $o/$m.time)" >> $D/done.txt; }
    t base  python3 $S/rss_sampler.py -i 0.005 -o $o/base.csv -- "${args[@]}"
    t opt1  env LD_PRELOAD=$D/mprshim.so MPR_MODE=info MPR_OUT=$o/opt1.csv "${args[@]}"
    t opt2a python3 $S/rss_sampler.py -i 0.005 -o $o/opt2a.csv -- env LD_PRELOAD=$D/mprshim.so MPR_MODE=trim "${args[@]}"
    t opt2b env MALLOC_TRIM_THRESHOLD_=0 MALLOC_TOP_PAD_=0 python3 $S/rss_sampler.py -i 0.005 -o $o/opt2b.csv -- "${args[@]}"
    [[ $prog == python ]] && t opt3 env PYTHONPATH=$HERE/tools/pysite MPR_PYOUT=$o/opt3.csv "${args[@]}"
    [[ $prog == lua ]] && t opt3 lua -e "local f=io.open('$o/opt3.csv','w') f:write('seconds,heap_bytes\n') debug.sethook(function() f:write(os.clock(),',',collectgarbage('count')*1024,'\n') f:flush() end,'',1000000)" "${args[@]:1}"
    t opt4  python3 $HERE/tools/refs_sampler.py -i 0.005 -w $(echo "$wall / 20" | bc -l) -o $o/opt4.csv -- "${args[@]}"
done; done
echo all-done
