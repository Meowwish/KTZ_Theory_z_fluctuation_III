"""Redraw manuscript Figure 1 from its archived field; no generator or statistics.

Run from any directory:
    python validation/replot_scalar_fields.py --output /tmp/figure1.png
The archived field is the seed-42 first realisation used by Block 5. For an
exact pixel comparison with the published PNG use Matplotlib 3.10.9.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def render(fields_path, output):
    with np.load(fields_path, allow_pickle=False) as data:
        X, Y, R_grid = data['X'], data['Y'], data['R_grid']
        Z, Z_bg, Z_res = data['Z'], data['Z_bg'], data['Z_res']
        disk_mask = data['disk_mask']
        analysis_R_in, analysis_R_out = data['analysis_R']
    assert np.array_equal(Z_res, np.where(disk_mask, Z-Z_bg, np.nan), equal_nan=True)
    # This archived grid contains centres on the original 15-kpc source edge.
    R_out = float(np.max(R_grid[disk_mask]))
    assert np.isclose(R_out, 15.0)
    Z_plot = np.where(disk_mask, Z, np.nan)
    dX_ax = Z_res
    ana = np.isfinite(dX_ax) & (R_grid >= analysis_R_in) & (R_grid <= analysis_R_out)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    vmin, vmax = np.nanpercentile(Z_plot, [1, 99])
    for ax, field, lab, title in [(axes[0], Z_plot, r'$\Sigma_X$', 'Simulation'),
                                 (axes[1], Z_bg, r'$\bar\mu$', 'Subtracted axisymmetric mean')]:
        im = ax.pcolormesh(X, Y, field, cmap='plasma', shading='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=lab); ax.set_title(title)
    lim = np.nanpercentile(np.abs(dX_ax[ana]), 99.5)
    im = axes[2].pcolormesh(X, Y, dX_ax, cmap='RdBu_r', shading='auto', vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axes[2], label=r'$\delta\Sigma_{X,{\rm ax}}=\Sigma_X-\bar\mu$')
    axes[2].set_title(r'Fluctuation used for $\xi$ (simulation $-\;\bar\mu$)')
    for rr in (analysis_R_in, analysis_R_out):
        axes[2].add_patch(plt.Circle((0, 0), rr, fill=False, ls='--', lw=0.8, color='0.25'))
    for ax in axes:
        ax.set_xlim(-R_out, R_out); ax.set_ylim(-R_out, R_out); ax.set_aspect('equal'); ax.set_xlabel(r'$x\;({\rm kpc})$')
    axes[0].set_ylabel(r'$y\;({\rm kpc})$')
    plt.tight_layout(); fig.savefig(output, dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, default=ROOT/'validation/outputs/fig1_realisation_fields.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'figs/validation_scalar_fields.png')
    args = parser.parse_args()
    render(args.fields, args.output)
