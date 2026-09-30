#!/bin/bash
. ../.venv/bin/activate
export THREADS=1
for s in 0 1 2; do for m in erm bnadapt tent arm xattn; do python icl_csida.py --proto $1 --method $m --seed $s; done; done
echo ALLDONE
