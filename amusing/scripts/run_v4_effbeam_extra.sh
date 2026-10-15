#!/bin/bash
# v4 extras (2026-10-14): effective-beam versions of the r<=5 kpc check (NGC 2466 S), the NGC 1084 S-calibration fit,
# and the phase-regression significances (null realisations built from the effective-beam L2 fits).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
L=logs/v4_effbeam_C.log; echo "start $(date)" >> $L
P66="scripts/paper_single_pointing.py --name NGC2466 --src-max 4.0 --rmin-fit 0.45"
P84="scripts/paper_single_pointing.py --name NGC1084 --rmin-fit 0.12"
V=scal_mockcov_fid0_v2_pixbeam
for e in "0.08 0.22" "0.22 0.35" "0.35 0.5"; do set -- $e
  [ -f outputs/NGC2466/ktz/$V/ngc2466_paper_nested_v3r5_ell$1-$2.json ] || python $P66 --variant $V --m1 1 --ell-min $1 --ell-max $2 --rmax-fit 5.0 --tag _v3r5_ell$1-$2 --figname zz_v4r5_$1 > logs/paper_ngc2466_${V}_v3r5_ell$1-$2.log 2>&1
done
python scripts/combine_split_evidence.py NGC2466 $V 0.08 0.22 0.22 0.35 0.35 0.5 v3r5 > logs/combine_ngc2466_${V}_v3r5.log 2>&1; echo "r5 $(date)" >> $L
# NGC 1084 S calibration with the effective beam
W=scal_mockcov_v2_pixbeam; d=outputs/NGC1084/ktz/$W; mkdir -p $d
for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz curves.npz; do [ -e outputs/NGC1084/ktz/scal_mockcov_v2/ngc1084_$x ] && ln -sf ../scal_mockcov_v2/ngc1084_$x $d/ngc1084_$x; done
python3 -c "
import json,math; m=json.load(open('outputs/NGC1084/ktz/scal_mockcov_v2/ngc1084_morphology.json')); m['effective_fwhm_kpc_parent']=m.get('effective_fwhm_kpc'); m['effective_fwhm_kpc']=math.sqrt(0.12**2+0.462*0.06**2); m['beam_variant']='pixbeam (effective Gaussian cell response)'; json.dump(m,open('$d/ngc1084_morphology.json','w'),indent=1)"
for e in "0.08 0.5" "0.5 3.2"; do set -- $e
  [ -f $d/ngc1084_paper_nested_v3_ell$1-$2.json ] || python $P84 --variant $W --m1 -71 --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname zz_v4s_$1 > logs/paper_ngc1084_${W}_v3_ell$1-$2.log 2>&1
done
python scripts/combine_split_evidence.py NGC1084 $W 0.08 0.5 0.5 3.2 v3 > logs/combine_ngc1084_${W}_v3.log 2>&1; echo "1084 S $(date)" >> $L
# wait for the main effective-beam fits, then the regression significances
until grep -q "ALL DONE" logs/v4_effbeam_A.log && grep -q "ALL DONE" logs/v4_effbeam_B.log; do sleep 60; done
python scripts/make_curve_from_split.py NGC1084 d16_mockcov_v2_pixbeam v3 L2 outputs/NGC1084/ktz/d16_mockcov_v2_pixbeam/ngc1084_curves_v3L2.npz
python scripts/mean_field_significance.py --variant outputs/NGC1084/ktz/d16_mockcov_v2_pixbeam --n 2000 --m1-control --curve outputs/NGC1084/ktz/d16_mockcov_v2_pixbeam/ngc1084_curves_v3L2.npz --out-suffix _v3m1 > logs/mf_ngc1084_pixbeam.log 2>&1
for c in scal d16; do V=${c}_mockcov_fid0_v2_pixbeam
  python scripts/make_curve_from_split.py NGC2466 $V v3 L2 outputs/NGC2466/ktz/$V/ngc2466_curves_v3L2.npz
  python scripts/mean_field_significance.py --variant outputs/NGC2466/ktz/$V --n 2000 --m1-control --curve outputs/NGC2466/ktz/$V/ngc2466_curves_v3L2.npz --out-suffix _v3m1 > logs/mf_ngc2466_${c}_pixbeam.log 2>&1
done
rm -f figs/zz_v4r5_* figs/zz_v4s_*
echo "ALL DONE $(date)" >> $L
