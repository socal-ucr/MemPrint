#!/bin/bash
# The rest of batch50.sh after its trace step, in order (the timing steps need the machine to themselves).
cd "$(dirname "$0")/.."; O=$HOME/memprint-data/2026-11-12-b50more; PY=$HOME/environments/memfp/bin/python
until grep -q "all-done trace" $O/done.txt; do sleep 30; done
dissertation/batch50_retry.sh > $O/retry.log 2>&1
echo "retry-done" >> $O/done.txt
PYTHONPATH=analysis $PY -W ignore dissertation/score.py b50more_build > $O/build.log 2>&1
dissertation/batch50.sh sample > $O/sample.log 2>&1
PYTHONPATH=analysis $PY -W ignore dissertation/score.py b50more > $O/score.log 2>&1
dissertation/batch50.sh cost > $O/cost.log 2>&1
dissertation/batch50.sh splittime > $O/splittime.log 2>&1
PYTHONPATH=analysis $PY -W ignore dissertation/buildtime.py > $O/buildtime.log 2>&1
echo "chain-done" >> $O/done.txt
