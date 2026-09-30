#!/bin/bash
. ../.venv/bin/activate
export THREADS=1
for s in 0 1 2; do
  python train.py --train ut_real  --target ut  --seed $s
  python train.py --train ntu_real --target ntu --seed $s
  python train.py --train ntu_real --target ut  --classes fall,run,walk --seed $s
  python train.py --train ut_real  --target ntu --classes fall,run,walk --seed $s
  python train.py --train ut_real  --target ut  --classes fall,run,walk --seed $s
  python train.py --train ntu_real --target ntu --classes fall,run,walk --seed $s
done
echo ALLDONE
