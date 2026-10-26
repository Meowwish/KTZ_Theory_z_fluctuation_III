#!/usr/bin/env python3
"""array_digest.py NPZ... : per-array sha256 (of the raw bytes) and sum/abs-max, to compare files across machines
whose metadata (paths, timestamps) differ."""
import sys, hashlib, numpy as np
for f in sys.argv[1:]:
    z = np.load(f)
    for k in sorted(z.files):
        a = z[k]
        if a.dtype.kind in "fiu":
            print(f.split("/")[-1], k, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16], f"{float(np.sum(a, dtype=np.float64)):.15e}")
        else:
            print(f.split("/")[-1], k, "non-numeric", hashlib.sha256(str(a).encode()).hexdigest()[:16])
