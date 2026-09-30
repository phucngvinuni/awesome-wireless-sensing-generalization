#!/bin/bash
# usage: run_sim.sh {ut|ntu}
. ../.venv/bin/activate
export THREADS=2
T=$1
until grep -q '^done' logs/gen_$T.log; do sleep 20; done
for s in 0 1 2; do
  for m in phys clean awgn unstructured; do
    python train.py --train sim --target $T --mode $m --seed $s
  done
done
for s in 0 1 2; do
  for m in phys clean; do
    python train.py --train sim --target $T --mode $m --classes fall,run,walk --seed $s
    python train.py --train sim --target $T --mode $m --norm sanitized --seed $s
  done
done
echo ALLDONE
