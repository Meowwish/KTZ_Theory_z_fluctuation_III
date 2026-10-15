"""Minimal read_cluster replacement for the two SLUG yield tables used by KT25 (cluster_yield FITS only)."""
import numpy as np
from astropy.io import fits
from types import SimpleNamespace
def read_cluster(stem):
    with fits.open(stem + '_cluster_yield.fits') as h:
        names = np.array([str(n).strip().title() for n in h[1].data['Name']])
        time = np.asarray(h[2].data['Time'], float)
        yld = np.asarray(h[2].data['Yield'], float)
    return SimpleNamespace(time=time, isotope_name=names, yld=yld)
