#!/bin/bash
# corpus_e1.sh's steps in order, after the hibits rerun and the truth runs (both under their own memory budgets).
E=$HOME/memprint-data/2026-11-23-corpus-e1; H=$HOME/memprint-data/2026-11-23-hibits
PY=$HOME/environments/memfp/bin/python; cd $HOME/MemPrint
until grep -q "^all-done" $H/done.txt && grep -q "^all-done truth" $E/done.txt; do sleep 60; done
dissertation/corpus_e1.sh split
PYTHONPATH=analysis $PY -W ignore dissertation/corpus_e1.py build > $E/build.log 2>&1
dissertation/corpus_e1.sh sample
PYTHONPATH=analysis $PY -W ignore dissertation/corpus_e1.py errors > $E/errors.log 2>&1
echo chain-done >> $E/done.txt
