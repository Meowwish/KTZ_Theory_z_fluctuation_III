#!/bin/bash
# v3 fiducial (2026-10-12): corrected covariance, m=1 nuisance marginalised in every model, sub-beam bins (r < 0.45 kpc) excluded,
# split prior in ell_star ([0.08,0.22],[0.22,0.35],[0.35,0.5]; plus [0.5,3.2] for the ell*-free rows), null and posterior-predictive covariances.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1
L=logs/v3_ngc2466.log; echo "start $(date)" >> $L
declare -A PH=( [scal]=1 [d16]=9 )
P="scripts/paper_single_pointing.py --name NGC2466 --src-max 4.0 --rmin-fit 0.45"
pieces() { # variant calib extra-args
  local V=$1 C=$2 EXTRA=$3
  for e in "0.08 0.22" "0.22 0.35" "0.35 0.5"; do set -- $e
    python $P --variant $V --m1 ${PH[$C]} --ell-min $1 --ell-max $2 --tag _v3_ell$1-$2 --figname ngc2466_v3_${1}_$V $EXTRA > logs/paper_ngc2466_${V}_v3_ell$1-$2.log 2>&1
  done
  python scripts/combine_split_evidence.py NGC2466 $V 0.08 0.22 0.22 0.35 0.35 0.5 v3 > logs/combine_ngc2466_${V}_v3.log 2>&1
}
mkvar() { # parent child  (symlink maps/cov/response; morphology copied)
  mkdir -p outputs/NGC2466/ktz/$2; for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz curves.npz; do [ -e outputs/NGC2466/ktz/$1/ngc2466_$x ] && ln -sf ../$1/ngc2466_$x outputs/NGC2466/ktz/$2/ngc2466_$x; done; cp outputs/NGC2466/ktz/$1/ngc2466_morphology.json outputs/NGC2466/ktz/$2/ngc2466_morphology.json
}
for c in scal d16; do
  # null covariance, fiducial
  pieces ${c}_mockcov_fid0_v2 $c ""; echo "$c null $(date)" >> $L
  # ell*-free extra piece
  python $P --variant ${c}_mockcov_fid0_v2 --m1 ${PH[$c]} --ell-min 0.5 --ell-max 3.2 --tag _v3_ell0.5-3.2 --figname ngc2466_v3_free_${c} > logs/paper_ngc2466_${c}_v3_ell0.5-3.2.log 2>&1; echo "$c ellfree $(date)" >> $L
  # posterior-predictive covariance from the v3 L2 fit (dominant piece), stochastic terms only
  python scripts/make_curve_from_split.py NGC2466 ${c}_mockcov_fid0_v2 v3 L2 outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_curves_v3L2.npz
  COMMON="--name NGC2466 --distance 73.8 --ra 116.3167 --dec -71.4100 --ba 0.95 --pa 212 --cov mock --pixel 0.12 --annulus 0.4 --bin 0.12 --rsep-max 8 --rmin 1.0 --rmax 10 --src-max 4.0 --ell-max 0.5"
  python scripts/amusing_ktz_stage.py $COMMON --calib $c --mock-curve outputs/NGC2466/ktz/${c}_mockcov_fid0_v2/ngc2466_curves_v3L2.npz --mock-stochastic --maps-only --tag _fid2_v3 > logs/ktz_ngc2466_${c}_fid2_v3.log 2>&1
  ln -sf ../${c}_mockcov_fid0_v2/ngc2466_window_response_all.npz outputs/NGC2466/ktz/${c}_mockcov_fid2_v3/ngc2466_window_response_all.npz
  pieces ${c}_mockcov_fid2_v3 $c ""; echo "$c pp $(date)" >> $L
  # geometry variants (null covariance, mock ell 0.6, r<=5 kpc; window responses from the v2 stage runs)
  for geo in g97 g90 g79; do pieces ${c}_mockcov_${geo}_lb_v2 $c ""; done; echo "$c geometry $(date)" >> $L
done
# S-calibration robustness: covariance settings, sampler, pixel term, pp recipe
for v in shrink20 shrink50 floor0 floor02 nmock150 nmock600 pixbeam; do pieces scal_mockcov_fid0_v2_$v scal ""; done; echo "cov variants $(date)" >> $L
mkvar scal_mockcov_fid0_v2 scal_mockcov_fid0_v2_seed3; pieces scal_mockcov_fid0_v2_seed3 scal "--seed 3"
mkvar scal_mockcov_fid0_v2 scal_mockcov_fid0_v2_nlive800; pieces scal_mockcov_fid0_v2_nlive800 scal "--nlive 800"; echo "sampler variants $(date)" >> $L
python scripts/make_curve_from_split.py NGC2466 scal_mockcov_fid0_v2 v3 L1n outputs/NGC2466/ktz/scal_mockcov_fid0_v2/ngc2466_curves_v3L1n.npz
COMMON="--name NGC2466 --distance 73.8 --ra 116.3167 --dec -71.4100 --ba 0.95 --pa 212 --cov mock --pixel 0.12 --annulus 0.4 --bin 0.12 --rsep-max 8 --rmin 1.0 --rmax 10 --src-max 4.0 --ell-max 0.5"
python scripts/amusing_ktz_stage.py $COMMON --calib scal --mock-curve outputs/NGC2466/ktz/scal_mockcov_fid0_v2/ngc2466_curves_v3L1n.npz --mock-stochastic --maps-only --tag _fid2L1n_v3 > logs/ktz_ngc2466_scal_fid2L1n_v3.log 2>&1
ln -sf ../scal_mockcov_fid0_v2/ngc2466_window_response_all.npz outputs/NGC2466/ktz/scal_mockcov_fid2L1n_v3/ngc2466_window_response_all.npz
pieces scal_mockcov_fid2L1n_v3 scal ""
python scripts/make_curve_from_split.py NGC2466 scal_mockcov_fid2_v3 v3 L2 outputs/NGC2466/ktz/scal_mockcov_fid2_v3/ngc2466_curves_v3L2pp.npz
python scripts/amusing_ktz_stage.py $COMMON --calib scal --mock-curve outputs/NGC2466/ktz/scal_mockcov_fid2_v3/ngc2466_curves_v3L2pp.npz --mock-stochastic --maps-only --tag _fid3_v3 > logs/ktz_ngc2466_scal_fid3_v3.log 2>&1
ln -sf ../scal_mockcov_fid0_v2/ngc2466_window_response_all.npz outputs/NGC2466/ktz/scal_mockcov_fid3_v3/ngc2466_window_response_all.npz
pieces scal_mockcov_fid3_v3 scal ""; echo "pp variants $(date)" >> $L
echo "ALL DONE $(date)" >> $L
