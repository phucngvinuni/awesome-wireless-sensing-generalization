#!/bin/bash
. ../.venv/bin/activate
rm -rf simdata/ut_H.npy simdata/ut_y.npy simdata/ut_meta.json simdata/ntu_*
THREADS=3 python gen.py ut 500 21 simdata/ut > logs/gen_ut_v3.log 2>&1
export THREADS=3
for s in 0 1 2; do
  for m in phys clean awgn unstructured; do
    python train.py --train sim --target ut --mode $m --norm sanitized --seed $s --tag v3
  done
done
for s in 0 1 2; do
  python train.py --train sim --target ut --mode phys --classes fall,run,walk --norm sanitized --seed $s --tag v3
  python train.py --train sim --target ut --mode clean --classes fall,run,walk --norm sanitized --seed $s --tag v3
done
echo ALLDONE
