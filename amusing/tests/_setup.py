import sys, numpy as np
from pathlib import Path
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN/"scripts"))
def setup(name, variant, m1, rmin, src, lo=None, hi=None, unified=True, cm=65, rmax=None, extra=()):
    argv = ["paper_single_pointing.py", "--name", name, "--variant", variant, "--m1", str(m1), "--rmin-fit", str(rmin), "--plot-only", "--src-max", str(src), "--cm-nodes", str(cm)]
    if lo is not None: argv += ["--ell-min", f"{lo:g}", "--ell-max", f"{hi:g}"]
    if rmax is not None: argv += ["--rmax-fit", str(rmax)]
    if unified: argv += ["--unified-response"]
    argv += list(extra)
    src_ = (RUN/"scripts"/"paper_single_pointing.py").read_text(); src_ = src_[:src_.index("NAMES = [")]
    g = {"__file__": str(RUN/"scripts"/"paper_single_pointing.py"), "__name__": "utest"}; old = sys.argv; sys.argv = argv
    try: exec(compile(src_, "psp_setup", "exec"), g)
    finally: sys.argv = old
    return g
