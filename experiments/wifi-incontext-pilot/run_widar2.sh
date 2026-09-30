#!/bin/bash
. ../.venv/bin/activate
export THREADS=2
for s in 0 1 2; do for m in arm xattn; do python icl_widar.py --proto $1 --method $m --seed $s; done; done
echo ALLDONE
