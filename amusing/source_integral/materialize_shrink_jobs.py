"""Relocate the archived sensitivity jobs to an existing source-integral workdir.

Dry-run by default; --execute creates inputs/jobs in a new directory, without
starting fits. Existing differing files are refused. SOURCE_WORK must contain
the accepted production tables and inputs produced by prepare_work.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_work', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--release', type=Path, default=ROOT/'amusing/release_20261118_shrink')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source, work, release = args.source_work.resolve(), args.workdir.resolve(), args.release.resolve()
    if work == source or source in work.parents:
        raise ValueError('Use a separate sensitivity workdir, not the production workdir')
    copies, jobs = {}, {}
    for path in sorted((release/'jobs').rglob('*.json')):
        job = json.loads(path.read_text())
        field = job['variant'].split('_sh')[0]
        galaxy = job['cfg']['g']
        variant = job['variant']
        inp = work/'inputs'/variant
        for item in (release/'inputs'/variant).iterdir():
            copies[inp/item.name] = item
        # The archived sensitivity inputs deliberately exclude raw/pixel maps.
        copies[inp/(galaxy+'_pixel_map.npz')] = source/'inputs'/(field+'_pixbeam_fa')/(galaxy+'_pixel_map.npz')
        job['cfg']['dir'] = str(inp)
        job['tables'] = str(source/'tables'/Path(job['tables']).name)
        job['fast'] = str(source/'tables'/Path(job['fast']).name)
        job['out'] = str(work/'runs'/Path(job['out']).name)
        for key in ('tables','fast'):
            if args.execute and not Path(job[key]).is_file():
                raise FileNotFoundError(f'Accepted existing table required: {job[key]}')
        jobs[work/'jobs'/variant/path.name] = json.dumps(job,indent=2)+'\n'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for dest, src in copies.items():
        if args.execute and not src.is_file():
            raise FileNotFoundError(src)
        if dest.exists() and (not src.exists() or sha(dest) != sha(src)):
            raise FileExistsError(f'Refusing to overwrite: {dest}')
    for dest, content in jobs.items():
        if dest.exists() and dest.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite: {dest}')
    if args.execute:
        for dest, src in copies.items():
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): shutil.copy2(src,dest)
        for dest, content in jobs.items():
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): dest.write_text(content)
        (work/'runs').mkdir(exist_ok=True)
    print(json.dumps(dict(mode='execute' if args.execute else 'dry-run', jobs=len(jobs), input_files=len(copies), workdir=str(work))))


if __name__ == '__main__':
    main()
