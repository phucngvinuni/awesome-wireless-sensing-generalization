#!/bin/bash
# usage: run_icl.sh {ut|ntu}
. ../.venv/bin/activate
export THREADS=2
for s in 0 1 2; do
  for m in erm erm_vd bnadapt tent arm xattn; do
    python icl.py --src $1 --method $m --seed $s --tag icl
  done
done
echo ALLDONE
