#!/bin/bash
# corpusmore.sh's steps in order; the timing steps (splittime, buildtime) last, with the machine to themselves.
cd "$(dirname "$0")/.."; O=$HOME/memprint-data/2026-11-13-corpusmore; PY=$HOME/environments/memfp/bin/python
dissertation/corpusmore.sh trace > $O/trace.log 2>&1
PYTHONPATH=analysis $PY -W ignore dissertation/score.py corpusmore_build > $O/build.log 2>&1
dissertation/corpusmore.sh sample > $O/sample.log 2>&1
PYTHONPATH=analysis $PY -W ignore dissertation/score.py corpusmore > $O/score.log 2>&1
dissertation/corpusmore.sh splittime > $O/splittime.log 2>&1
PYTHONPATH=analysis $PY -W ignore dissertation/buildtime.py corpus > $O/buildtime.log 2>&1
echo "chain-done" >> $O/done.txt
