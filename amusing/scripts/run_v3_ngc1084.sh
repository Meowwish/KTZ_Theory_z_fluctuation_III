#!/bin/bash
# v3 fiducial for NGC 1084: corrected covariance, m=1 nuisance (phase -87 deg D16, -71 S), first bin (r < 0.12 kpc) excluded, split prior [0.08,0.5],[0.5,3.2].
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
L=logs/v3_ngc1084.log; echo "start $(date)" >> $L
declare -A PH=( [scal]=-71 [d16]=-87 )
P="scripts/paper_single_pointing.py --name NGC1084 --rmin-fit 0.12"
pieces() { local V=$1 C=$2 EXTRA=$3
  for e in "0.08 0.5" "0.5 3.2"; do set -- $e
    python $P --variant $V --m1 ${PH[$C]} --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname ngc1084_v3_${1}_$V $EXTRA > logs/paper_ngc1084_${V}_v3_ell$1-$2.log 2>&1
  done
  python scripts/combine_split_evidence.py NGC1084 $V 0.08 0.5 0.5 3.2 v3 > logs/combine_ngc1084_${V}_v3.log 2>&1
}
mkvar() { mkdir -p outputs/NGC1084/ktz/$2; for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz curves.npz; do [ -e outputs/NGC1084/ktz/$1/ngc1084_$x ] && ln -sf ../$1/ngc1084_$x outputs/NGC1084/ktz/$2/ngc1084_$x; done; cp outputs/NGC1084/ktz/$1/ngc1084_morphology.json outputs/NGC1084/ktz/$2/ngc1084_morphology.json; }
N1="--name NGC1084 --distance 19.7 --ba 0.918 --cov mock --mock-ell 0.35"
pieces d16_mockcov_v2 d16 ""; echo "d16 null $(date)" >> $L
python scripts/make_curve_from_split.py NGC1084 d16_mockcov_v2 v3 L2 outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_curves_v3L2.npz
python scripts/amusing_ktz_stage.py $N1 --calib d16 --mock-curve outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_curves_v3L2.npz --maps-only --tag _v3_pp > logs/ktz_ngc1084_d16_v3_pp.log 2>&1
ln -sf ../d16_mockcov_v2/ngc1084_window_response_all.npz outputs/NGC1084/ktz/d16_mockcov_v3_pp/ngc1084_window_response_all.npz
pieces d16_mockcov_v3_pp d16 ""; echo "d16 pp $(date)" >> $L
pieces scal_mockcov_v2 scal ""; echo "scal null $(date)" >> $L
pieces d16_mockcov_t05_v2 d16 ""; echo "t05 $(date)" >> $L
for v in shrink20 shrink50 floor0 floor02 nmock150 nmock600 pixbeam; do pieces d16_mockcov_v2_$v d16 ""; done; echo "cov variants $(date)" >> $L
pieces d16_mockcov_v2_seed3 d16 "--seed 3"; pieces d16_mockcov_v2_nlive800 d16 "--nlive 800"; echo "sampler variants $(date)" >> $L
python scripts/make_curve_from_split.py NGC1084 d16_mockcov_v2 v3 L1n outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_curves_v3L1n.npz
python scripts/amusing_ktz_stage.py $N1 --calib d16 --mock-curve outputs/NGC1084/ktz/d16_mockcov_v2/ngc1084_curves_v3L1n.npz --maps-only --tag _v3_ppL1n > logs/ktz_ngc1084_d16_v3_ppL1n.log 2>&1
ln -sf ../d16_mockcov_v2/ngc1084_window_response_all.npz outputs/NGC1084/ktz/d16_mockcov_v3_ppL1n/ngc1084_window_response_all.npz
pieces d16_mockcov_v3_ppL1n d16 ""
python scripts/make_curve_from_split.py NGC1084 d16_mockcov_v3_pp v3 L2 outputs/NGC1084/ktz/d16_mockcov_v3_pp/ngc1084_curves_v3L2pp.npz
python scripts/amusing_ktz_stage.py $N1 --calib d16 --mock-curve outputs/NGC1084/ktz/d16_mockcov_v3_pp/ngc1084_curves_v3L2pp.npz --maps-only --tag _v3_pp2 > logs/ktz_ngc1084_d16_v3_pp2.log 2>&1
ln -sf ../d16_mockcov_v2/ngc1084_window_response_all.npz outputs/NGC1084/ktz/d16_mockcov_v3_pp2/ngc1084_window_response_all.npz
pieces d16_mockcov_v3_pp2 d16 ""; echo "pp variants $(date)" >> $L
echo "ALL DONE $(date)" >> $L
