"""Materialize released inputs and source-table configurations in a new workdir.

Usage: python amusing/source_integral/prepare_work.py WORKDIR [--execute]
Dry-run by default. Existing non-identical files are refused. No fit is started.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/'amusing/release_20261026'
ALIASES = dict(S='S_pixbeam_fa', S_r5='S_pixbeam_fa', D16='D16_pixbeam_fa',
               N1084D16='N1084D16_pixbeam_fa', N1084S='N1084S_pixbeam_fa',
               N1084t05='N1084t05_pixbeam_fa', S_EB2='S_ppe_si1026',
               S_r5_EB2='S_ppe_si1026', D16_EB2='D16_ppe_si1026',
               N1084D16_EB2='N1084D16_ppe_si1026')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    work = args.workdir.resolve()
    index = json.loads((RELEASE/'metadata/input_index.json').read_text())
    plan = {}
    for key, item in index.items():
        g = item['galaxy'].lower()
        dest = work/'inputs'/ALIASES[key]
        for suffix in ('morphology.json', 'diagnostics.csv', 'pixel_map.npz', 'fit_covariance.npz'):
            source = (ROOT/item['pixel_map'] if suffix == 'pixel_map.npz' else
                      ROOT/item['covariance'] if suffix == 'fit_covariance.npz' else
                      ROOT/item['input_dir']/f'{g}_{suffix}')
            target = dest/f'{g}_{suffix}'
            if target in plan:
                assert source.read_bytes() == plan[target].read_bytes()
            plan[target] = source
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for target, source in plan.items():
        assert source.exists(), source
        if target.exists() and sha(target) != sha(source):
            raise FileExistsError(f'Refusing to overwrite different input: {target}')
        if args.execute:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    for source in sorted((RELEASE/'configs/tables').glob('*.json')):
        cfg = json.loads(source.read_text())
        cfg['dir'] = str(work/'inputs'/Path(cfg['dir']).name)
        cfg['out'] = str(work/'tables'/source.stem)
        target = work/'tables'/(source.stem+'.json')
        content = json.dumps(cfg, indent=2)+'\n'
        if target.exists() and target.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite different config: {target}')
        if args.execute:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    print(json.dumps(dict(mode='execute' if args.execute else 'dry-run',
                          workdir=str(work), input_files=len(plan), table_configs=5)))

if __name__ == '__main__':
    main()
