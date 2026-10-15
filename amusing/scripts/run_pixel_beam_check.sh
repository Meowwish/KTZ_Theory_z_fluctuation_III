#!/bin/bash
# Pixel-averaging / beam-convention robustness check for the paper fits (2026-10-10).
# Each variant reuses the maps, window responses and covariance of the parent variant and
# changes only effective_fwhm_kpc in the morphology file, which is the beam the model kernels see.
# Under the code convention (kernel variance += 0.5*(FWHM/2.3548)^2):
#   pixbeam  : adds the exact square-pixel term dx^2/12 to the kernel variance
#              -> FWHM_eff^2 = FWHM^2 + 2.3548^2 dx^2/6
#   fullbeam : kernel variance += (FWHM/2.3548)^2, i.e. the full beam variance (factor-2 question)
#   fullpix  : full beam + pixel term
cd /home/claude/amusing
mk() { # name parent child fwhm_eff
  local g=$1 p=$2 c=$3 f=$4; local lg=$(echo $g | tr A-Z a-z)
  local src=outputs/$g/ktz/$p dst=outputs/$g/ktz/$c; mkdir -p $dst
  for x in pixel_map.npz diagnostics.csv fit_covariance.npz window_response_all.npz; do ln -sf ../$p/${lg}_$x $dst/${lg}_$x; done
  python3 - "$src/${lg}_morphology.json" "$dst/${lg}_morphology.json" "$f" "$c" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); m['effective_fwhm_kpc_parent']=m.get('effective_fwhm_kpc', m['target_fwhm_kpc']); m['effective_fwhm_kpc']=float(sys.argv[3]); m['beam_variant']=sys.argv[4]
json.dump(m,open(sys.argv[2],'w'),indent=1)
PY
}
py() { python3 -c "import math; print($1)"; }
# NGC 1084: FWHM 0.12, dx 0.06
mk NGC1084 d16_mockcov d16_mockcov_pixbeam  $(py "math.sqrt(0.12**2+2.354820045**2*0.06**2/6)")
mk NGC1084 d16_mockcov d16_mockcov_fullbeam $(py "math.sqrt(2)*0.12")
mk NGC1084 d16_mockcov d16_mockcov_fullpix  $(py "math.sqrt(2*0.12**2+2.354820045**2*0.06**2/6)")
# NGC 2466: FWHM 0.44796, dx 0.12
F=0.4479562058168646
mk NGC2466 scal_mockcov_fid0 scal_mockcov_fid0_pixbeam  $(py "math.sqrt($F**2+2.354820045**2*0.12**2/6)")
mk NGC2466 d16_mockcov_fid0  d16_mockcov_fid0_pixbeam   $(py "math.sqrt($F**2+2.354820045**2*0.12**2/6)")
mk NGC2466 scal_mockcov_fid0 scal_mockcov_fid0_fullbeam $(py "math.sqrt(2)*$F")
mk NGC2466 d16_mockcov_fid0  d16_mockcov_fid0_fullbeam  $(py "math.sqrt(2)*$F")
for v in d16_mockcov_pixbeam d16_mockcov_fullbeam d16_mockcov_fullpix; do
  python scripts/paper_single_pointing.py --name NGC1084 --variant $v --figname ngc1084_single_pointing_$v > logs/paper_ngc1084_$v.log 2>&1; echo "NGC1084 $v $(date)" >> logs/pixel_beam_check.log
done
for v in scal_mockcov_fid0_pixbeam d16_mockcov_fid0_pixbeam scal_mockcov_fid0_fullbeam d16_mockcov_fid0_fullbeam; do
  python scripts/paper_single_pointing.py --name NGC2466 --variant $v --ell-max 0.5 --src-max 4.0 --figname ngc2466_single_pointing_$v > logs/paper_ngc2466_$v.log 2>&1; echo "NGC2466 $v $(date)" >> logs/pixel_beam_check.log
done
echo "ALL DONE $(date)" >> logs/pixel_beam_check.log
