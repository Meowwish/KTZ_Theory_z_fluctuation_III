"""Prepare archived common-covariance sensitivity fits; no computation is started.

Usage: python amusing/source_integral/prepare_pps_work.py WORKDIR [--execute]
Dry-run by default; existing differing files are refused. Response tables must
be built separately with the existing source-integral release instructions.
"""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/'amusing/release_20261030_pps'
FIELDS = {
    'S': ('ngc2466', 'S_pixbeam_fa', 'N2466S', 1., .45, 4.,
          [[.08,.22],[.22,.35],[.35,.5],[.5,3.2]]),
    'D16': ('ngc2466', 'D16_pixbeam_fa', 'N2466D16', 9., .45, 4.,
            [[.08,.22],[.22,.35],[.35,.5],[.5,3.2]]),
    'N1084D16': ('ngc1084', 'N1084D16_pixbeam_fa', 'N1084D16', -87., .12, 2.5,
                 [[.08,.5],[.5,3.2]]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workdir', type=Path)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    work = args.workdir.resolve()
    copies, generated = {}, {}
    for field, (gal, base, tab, phase, cut, srcmax, pieces) in FIELDS.items():
        source = ROOT/'amusing/release_20261022/stage_inputs/inputs'/base
        for suffix in ('pixel_map.npz','diagnostics.csv','morphology.json','fit_covariance.npz'):
            copies[work/'inputs'/base/f'{gal}_{suffix}'] = source/f'{gal}_{suffix}'
        cfg = json.loads((RELEASE/f'configs/{field}.json').read_text())
        cfg.update(dir=str(work/'inputs'/base), out=str(work/'tables'/tab))
        generated[work/f'configs/{field}.json'] = cfg
        for reference in ('L2gen','L0gen'):
            variant = f'{field}_{reference}'
            fit_input = work/'inputs'/variant
            for suffix in ('pixel_map.npz','diagnostics.csv','morphology.json'):
                copies[fit_input/f'{gal}_{suffix}'] = source/f'{gal}_{suffix}'
            copies[fit_input/f'{gal}_fit_covariance.npz'] = RELEASE/f'covariances/{variant}.npz'
            for lo, hi in pieces:
                for model in ('L0q','L2q'):
                    tag = f'v3_ell{lo:g}-{hi:g}'
                    name = f'{variant}__{tag}__{model}'
                    generated[work/'jobs'/f'{name}.json'] = dict(
                        out=str(work/'runs'/name),
                        cfg=dict(dir=str(fit_input), g=gal, m1=phase,
                                 rmin_fit=cut, src_max=srcmax, ell_min=lo, ell_max=hi),
                        tables=str(work/'tables'/f'{tab}_tables.npz'),
                        fast=str(work/'tables'/f'{tab}_fast_{base}.npz'),
                        model=model, nlive=400, seed=2, checkpoint_every=15,
                        eps_prior='manuscript', variant=variant, tag=tag,
                        priority=0 if model=='L0q' else 2)
    for target, source in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise FileExistsError(f'Refusing to overwrite differing file: {target}')
    for target, value in generated.items():
        content = json.dumps(value, indent=2)+'\n'
        if target.exists() and target.read_text()!=content:
            raise FileExistsError(f'Refusing to overwrite differing file: {target}')
    if args.execute:
        for target, source in copies.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source,target)
        for target, value in generated.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(mode='execute' if args.execute else 'dry-run',
                          workdir=str(work), copied_files=len(copies),
                          generation_configs=3, fit_jobs=40)))


if __name__=='__main__':
    main()
