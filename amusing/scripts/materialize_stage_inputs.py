"""Copy the four archived stage files to a new work directory; never calculate a fit.

The plan is printed by default. --execute copies only missing or byte-identical
files; a non-identical destination is refused, not overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

STAGE=Path(__file__).resolve().parents[1]/'release_20261022/stage_inputs'

def materialize(case,directory,execute=False):
    mapping=json.loads((STAGE/'case_inputs.json').read_text())['cases']
    c=mapping[case];directory=Path(directory)
    plan=[]
    for suffix,name in c['inputs'].items():
        src=STAGE/name;dst=directory/src.name
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if digest!=c['input_sha256'][suffix]:raise ValueError(f'Archive checksum mismatch: {src}')
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=digest:
            raise FileExistsError(f'Refusing to overwrite a different file: {dst}')
        plan.append((src,dst))
    for src,dst in plan:
        print(f'{src} -> {dst}')
    if execute:
        directory.mkdir(parents=True,exist_ok=True)
        for src,dst in plan:
            if not dst.exists():shutil.copyfile(src,dst)
    return c

if __name__=='__main__':
    choices=json.loads((STAGE/'case_inputs.json').read_text())['cases']
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',required=True,choices=choices)
    ap.add_argument('--directory',required=True,type=Path)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();materialize(a.case,a.directory,a.execute)
