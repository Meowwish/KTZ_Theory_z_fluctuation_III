#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
run() { [ -f "$1" ] || python scripts/mean_field_significance.py "${@:2}" > /dev/null 2>&1; }
run outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_mean_field_significance_v3m1.json --variant outputs/NGC1084/ktz/d16_mockcov_v2 --n 2000 --m1-control --curve outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_curves_v3L2.npz --out-suffix _v3m1
for c in scal d16; do
  run outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_mean_field_significance_v3m1.json --variant outputs/NGC2466/ktz/${c}_mockcov_fid0_v2 --n 2000 --m1-control --curve outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_curves_v3L2.npz --out-suffix _v3m1
  run outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_mean_field_significance_v3.json --variant outputs/NGC2466/ktz/${c}_mockcov_fid0_v2 --n 2000 --curve outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_curves_v3L2.npz --out-suffix _v3
  for p in P1 P2 P3 P4; do pl=$(echo $p | tr A-Z a-z)
    run outputs/NGC2466_$p/ktz/${c}_mockcov/ngc2466_${pl}_mean_field_significance_v3m1.json --variant outputs/NGC2466_$p/ktz/${c}_mockcov --n 2000 --m1-control --curve outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_curves_v3L2.npz --out-suffix _v3m1
  done
done
echo "MF DONE $(date)" >> logs/v3_ngc2466.log
