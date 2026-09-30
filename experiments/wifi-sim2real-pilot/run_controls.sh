#!/bin/bash
. ../.venv/bin/activate
export THREADS=3
for s in 0 1 2; do
  for m in awgn unstructured; do
    python train.py --train sim --target ut --mode $m --classes fall,run,walk --norm sanitized --seed $s --tag v3
  done
done
echo ALLDONE
