#!/bin/bash
# v4 (2026-10-14): effective Gaussian response for finite cells as the fiducial model beam,
# FWHM_eff^2 = FWHM_b^2 + 0.462 dx^2 (127 pc NGC 1084, 455 pc NGC 2466). Same data vectors, covariances (null and
# posterior-predictive), window responses, m=1 nuisance, sub-beam cut and split ell* prior as v3; only
# effective_fwhm_kpc in the morphology file changes. Outputs carry tag v3 inside *_pixbeam variant directories.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
WHICH=$1   # A or B
L=logs/v4_effbeam_$WHICH.log; echo "start $(date)" >> $L
mkpix() { # galaxy parent child fwhm
  local G=$1 P=$2 C=$3 F=$4; local lg=$(echo $G | tr A-Z a-z); local d=outputs/$G/ktz/$C; mkdir -p $d
  for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz curves.npz; do [ -e outputs/$G/ktz/$P/${lg}_$x ] && ln -sf ../$P/${lg}_$x $d/${lg}_$x; done
  python3 - "$d/${lg}_morphology.json" "outputs/$G/ktz/$P/${lg}_morphology.json" "$F" <<'PY'
import json,sys
m=json.load(open(sys.argv[2])); m['effective_fwhm_kpc_parent']=m.get('effective_fwhm_kpc'); m['effective_fwhm_kpc']=float(sys.argv[3]); m['beam_variant']='pixbeam (effective Gaussian cell response)'
json.dump(m,open(sys.argv[1],'w'),indent=1)
PY
}
F84=$(python3 -c "import math;print(math.sqrt(0.12**2+0.462*0.06**2))")
F66=$(python3 -c "import math;print(math.sqrt(0.4479562058168646**2+0.462*0.12**2))")
declare -A PH=( [scal]=1 [d16]=9 )
P66="scripts/paper_single_pointing.py --name NGC2466 --src-max 4.0 --rmin-fit 0.45"
P84="scripts/paper_single_pointing.py --name NGC1084 --rmin-fit 0.12"
pieces66() { local V=$1 C=$2
  for e in "0.08 0.22" "0.22 0.35" "0.35 0.5"; do set -- $e
    [ -f outputs/NGC2466/ktz/$V/ngc2466_paper_nested_v3_ell$1-$2.json ] || python $P66 --variant $V --m1 ${PH[$C]} --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname zz_v4_${1}_$V > logs/paper_ngc2466_${V}_v3_ell$1-$2.log 2>&1
  done
  python scripts/combine_split_evidence.py NGC2466 $V 0.08 0.22 0.22 0.35 0.35 0.5 v3 > logs/combine_ngc2466_${V}_v3.log 2>&1
}
if [ "$WHICH" = A ]; then CALS="scal"; else CALS="d16"; fi
for c in $CALS; do
  V=${c}_mockcov_fid0_v2_pixbeam; mkpix NGC2466 ${c}_mockcov_fid0_v2 $V $F66
  pieces66 $V $c; echo "$c null $(date)" >> $L
  [ -f outputs/NGC2466/ktz/$V/ngc2466_paper_nested_v3_ell0.5-3.2.json ] || python $P66 --variant $V --m1 ${PH[$c]} --ell-min 0.5 --ell-max 3.2 --tag _v3_ell0.5-3.2 --figname zz_v4_free_$c > logs/paper_ngc2466_${V}_v3_ell0.5-3.2.log 2>&1
  python scripts/combine_split_evidence.py NGC2466 $V 0.08 0.22 0.22 0.35 0.35 0.5 0.5 3.2 v3:free > logs/combine_ngc2466_${V}_v3free.log 2>&1; echo "$c free $(date)" >> $L
  W=${c}_mockcov_fid2_v3_pixbeam; mkpix NGC2466 ${c}_mockcov_fid2_v3 $W $F66
  pieces66 $W $c; echo "$c pp $(date)" >> $L
done
if [ "$WHICH" = A ]; then
  W=d16_mockcov_v3_pp_pixbeam; mkpix NGC1084 d16_mockcov_v3_pp $W $F84
  for e in "0.08 0.5" "0.5 3.2"; do set -- $e
    [ -f outputs/NGC1084/ktz/$W/ngc1084_paper_nested_v3_ell$1-$2.json ] || python $P84 --variant $W --m1 -87 --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname zz_v4_1084_$1 > logs/paper_ngc1084_${W}_v3_ell$1-$2.log 2>&1
  done
  python scripts/combine_split_evidence.py NGC1084 $W 0.08 0.5 0.5 3.2 v3 > logs/combine_ngc1084_${W}_v3.log 2>&1; echo "1084 pp $(date)" >> $L
fi
rm -f figs/zz_v4_*
echo "ALL DONE $(date)" >> $L
