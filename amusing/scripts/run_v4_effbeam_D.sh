#!/bin/bash
# v4 extra (2026-10-14): NGC 1084 D16 with the 0.5-kpc H-alpha template, effective-beam response (template-scale check).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
L=logs/v4_effbeam_D.log; echo "start $(date)" >> $L
until grep -q "1084 S" logs/v4_effbeam_C.log; do sleep 60; done
W=d16_mockcov_t05_v2_pixbeam; d=outputs/NGC1084/ktz/$W; mkdir -p $d
for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz curves.npz; do [ -e outputs/NGC1084/ktz/d16_mockcov_t05_v2/ngc1084_$x ] && ln -sf ../d16_mockcov_t05_v2/ngc1084_$x $d/ngc1084_$x; done
python3 -c "
import json,math; m=json.load(open('outputs/NGC1084/ktz/d16_mockcov_t05_v2/ngc1084_morphology.json')); m['effective_fwhm_kpc_parent']=m.get('effective_fwhm_kpc'); m['effective_fwhm_kpc']=math.sqrt(0.12**2+0.462*0.06**2); m['beam_variant']='pixbeam (effective Gaussian cell response)'; json.dump(m,open('$d/ngc1084_morphology.json','w'),indent=1)"
for e in "0.08 0.5" "0.5 3.2"; do set -- $e
  [ -f $d/ngc1084_paper_nested_v3_ell$1-$2.json ] || python scripts/paper_single_pointing.py --name NGC1084 --rmin-fit 0.12 --variant $W --m1 -87 --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname zz_v4t_$1 > logs/paper_ngc1084_${W}_v3_ell$1-$2.log 2>&1
done
python scripts/combine_split_evidence.py NGC1084 $W 0.08 0.5 0.5 3.2 v3 > logs/combine_ngc1084_${W}_v3.log 2>&1
rm -f figs/zz_v4t_*
echo "ALL DONE $(date)" >> $L
