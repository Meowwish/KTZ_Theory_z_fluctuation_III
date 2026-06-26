"""
Script to make all the plots in Krumholz, Ting, Li, Zhang, Mead, & Ness (2025)
"""

import matplotlib.pyplot as plt
import numpy as np
from slugpy import read_cluster
from scipy.stats import pearsonr
import astropy.units as u
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import os.path as osp

# Make directory into which to dump outputs
try:
    os.mkdir('output')
except FileExistsError:
    pass

# Grab Asplund (2009) abundances and Seitenzahl+ (2013) type Ia yields
from asplund09_abd import solar_abd
from seitenzahl13_yld import s13_yld

# Create dict for each element, containing symbol and initial fraction
prod = { s['sym'] : { 'f0' : s['frac'], 'Z' : s['Z'] }
         for s in solar_abd }

# Read slug yields for SNII and AGB
slug_agb = read_cluster(osp.join('slug', 'yields_agb'))
slug_snii = read_cluster(osp.join('slug', 'yields_snii'))
time = slug_agb.time * u.yr

# List of r-process-dominated elements to skip, going up to Z = 70
r_process = ['Ru', 'Rh', 'Rd', 'Ag', 'Cd', 'In', 'Sb', 'Te', 'I', 'Xe',
             'Cs', 'La', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
             'Ho', 'Er', 'Tm', 'Yb']

# List of elements to include
Zlo = 6   # C
Zhi = 60  # Nd
elem_for_table = [ s['sym'] for s in solar_abd
                   if (s['Z'] >= Zlo and s['Z'] <= Zhi)
                   and not s['sym'] in r_process]

# Get number of SNIa at each time using the DTD of Maoz & Graur
# (2017), for which SNIa rate is dN/dt = norm * (t/tmin)^alpha for t
# in [tmin, tmax] 
tmin = 40 * u.Myr      # Minimum delay time
tmax = 13.7 * u.Gyr    # Maximum delay time
alpha = -1.1           # Slope
norm = 1.3e-3 * (1 + alpha) / (tmax * (tmax/tmin)**alpha - tmin)
   # Normalization set to give 1.3e-3 SNIa per Msun of stars formed
   # from tmin to tmax
nIa = norm * (time * (time/tmin)**alpha - tmin) / (1 + alpha)
nIa[time < tmin] = 0

# Group slug yields by element, summing over isotope
for e in prod.keys():

    # Yields summing over all processes, and differentiated by process
    yld_tot = np.zeros(time.size)
    yld_snii = np.zeros(time.size)
    yld_agb = np.zeros(time.size)
    yld_snia = np.zeros(time.size)

    # SNII
    idx = np.where(slug_snii.isotope_name == e)[0]
    for i in idx:
        yld_snii += slug_snii.yld[:,i]
        yld_tot += slug_snii.yld[:,i]

    # AGB
    idx = np.where(slug_agb.isotope_name == e)[0]
    for i in idx:
        yld_agb += slug_agb.yld[:,i]
        yld_tot += slug_agb.yld[:,i]
    
    # Seitenzahl yields (type Ia)
    idx = np.where(np.array(s13_yld['sym']) == e)[0]
    for i in idx:
        yld_tot += s13_yld['ylds']['N100'][i] * nIa
        yld_snia += s13_yld['ylds']['N100'][i] * nIa
        
    # Store yields
    prod[e]['yld_tot'] = yld_tot
    prod[e]['yld_snii'] = yld_snii
    prod[e]['yld_snia'] = yld_snia
    prod[e]['yld_agb'] = yld_agb

# Get total mass return over all elements
mass_return = np.sum(
    np.array( [ prod[k]['yld_tot'] for k in prod.keys() ] ),
    axis=0)

# Get net production, 50% production time, and total fractional
# contribution for all elements and all processes
for e in prod.keys():
    prod[e]['net_tot'] = prod[e]['yld_tot'] - mass_return * prod[e]['f0']
    prod[e]['net_snii'] = prod[e]['yld_snii'] - mass_return * prod[e]['f0']
    prod[e]['net_snia'] = prod[e]['yld_snia'] - mass_return * prod[e]['f0']
    prod[e]['net_agb'] = prod[e]['yld_agb'] - mass_return * prod[e]['f0']
    prod[e]['t50_tot'] = time[ np.argmax(
        prod[e]['net_tot'] > 0.5 * prod[e]['net_tot'][-1]
        )]
    prod[e]['t50_snii'] = time[ np.argmax(
        prod[e]['yld_snii'] > 0.5 * prod[e]['yld_snii'][-1]
        )]
    prod[e]['t50_snia'] = time[ np.argmax(
        prod[e]['yld_snia'] > 0.5 * prod[e]['yld_snia'][-1]
        )]
    prod[e]['t50_agb'] = time[ np.argmax(
        prod[e]['yld_agb'] > 0.5 * prod[e]['yld_agb'][-1]
        )]
    if prod[e]['yld_tot'][-1] > 0:
        prod[e]['f_snii'] = prod[e]['yld_snii'][-1] / prod[e]['yld_tot'][-1]
        prod[e]['f_snia'] = prod[e]['yld_snia'][-1] / prod[e]['yld_tot'][-1]
        prod[e]['f_agb'] = prod[e]['yld_agb'][-1] / prod[e]['yld_tot'][-1]
        if (prod[e]['f_snii'] > prod[e]['f_snia']) and \
           (prod[e]['f_snii'] > prod[e]['f_agb']):
            prod[e]['main'] = 'snii'
        elif (prod[e]['f_snia'] > prod[e]['f_snii']) and \
             (prod[e]['f_snia'] > prod[e]['f_agb']):
            prod[e]['main'] = 'snia'
        else:
            prod[e]['main'] = 'agb'
    
# List of elements for plot of delay times, grouped by dominant source
elem_to_plot = { 'SNII-Wind' : ['C', 'O', 'S'],
                 'AGB' : ['N', 'Sr', 'Ba'],
                 'SNIa' : [ 'V', 'Mn', 'Fe' ]
                 }

# Plot of delay times
fig = plt.figure(1, figsize=(4,3))
fig.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
ax = fig.add_subplot(1,1,1)
ls = ['-', '--', '-.']
for i, grp in enumerate(elem_to_plot.keys()):
    clr = 'C{:d}'.format(i)
    for j, e in enumerate(elem_to_plot[grp]):
        ax.plot(time.to(u.Myr),
                prod[e]['net_tot'] / prod[e]['net_tot'][-1],
                label=r"{:s} ({:d})".
                format(e, int(round(prod[e]['t50_tot'].
                                    to(u.Myr).value))),
                color=clr, ls=ls[j])
ax.set_xscale('log')
ax.set_xlabel(r'$t$ [Myr]')
ax.set_ylabel('$p_X(t)/p_X(10\\mbox{ Gyr})$')
ax.set_xlim([3,1e4])
ax.set_ylim([0,1.])
ax.legend(prop={'size': 8}, ncol=1, loc='lower right')
plt.subplots_adjust(left=0.15, bottom=0.17, top=0.95, right=0.95)

# Save plot of delay production showing delay times
plt.savefig(osp.join('output', 'delay_time.pdf'))

# List of injection channels
channels = ['snii', 'agb', 'snia']

# Generate latex-formatted table
fp = open(osp.join("output", "element_summary.tex"), "w")
for i, e in enumerate(elem_for_table):
    print("{:2d} ({:s}) ".format(prod[e]['Z'], e), file=fp)
    for src in channels:
        if prod[e]['f_'+src] > 0.005:
            tstr = "{:4.1f}".format(prod[e]['t50_'+src].
                                    to(u.Myr).value).strip()
            if len(tstr) < 5:
                tstr = "\\phantom{" + "0"*(5-len(tstr)) + "}"+tstr
        else:
            tstr = "-"
        print("& {:4.2f}".format(prod[e]['f_'+src]) + " & " +
                 tstr, file=fp)
    print("\\\\", file=fp)
fp.close()

# Set fiducial physical and observational parameters
lcorr = 1 * u.kpc
h = 150 * u.pc
sigmag = 7 * u.km/u.s
kappa = h * sigmag / 3
sigmav = 2 * u.km / u.s
sigma_inj = { 'snii' : 60 * u.pc,
              'snia' : 60 * u.pc,
              'agb' : 2 * u.pc
             }
sigmaw2 = 120
Gamma = 3e-6 / u.Myr / u.pc**2  # Cluster formation rate
FWHM_beam = 20 * u.pc           # Sample beam size
sigma_beam = FWHM_beam / (2**1.5 * np.log(2))
fgas = 2                        # Gas error
sigma_logZ = [0.01, 0.03]       # Stellar metallicity errors

# Alternative values for select parameters
kappa_alt = np.array([1/3., 3.]) * kappa
sigmav_alt = np.array([1/3., 3.]) * sigmav
lcorr_alt = np.array([1/2., 2.]) * lcorr

# Define a function to evaluate the non-normalized elemental
# cross-correlation function
def computeXi(p1, p2, beam = False, return_coef = False, use_alt = False):
    Xi = 0
    alpha = np.zeros((len(channels), len(channels)))
    beta = np.zeros((len(channels), len(channels)))
    gamma = np.zeros((len(channels), len(channels)))
    if use_alt:
        Xi_kappa_alt = 0
        Xi_sigmav_alt = 0
        Xi_lcorr_alt = 0
        alpha_kappa_alt = np.zeros((len(channels), len(channels), 2))
        alpha_sigmav_alt = np.zeros((len(channels), len(channels), 2))
        alpha_lcorr_alt = np.zeros((len(channels), len(channels), 2))
        beta_kappa_alt = np.zeros((len(channels), len(channels), 2))
        beta_sigmav_alt = np.zeros((len(channels), len(channels), 2))
        beta_lcorr_alt = np.zeros((len(channels), len(channels), 2))
        gamma_kappa_alt = np.zeros((len(channels), len(channels), 2))
        gamma_sigmav_alt = np.zeros((len(channels), len(channels), 2))
        gamma_lcorr_alt = np.zeros((len(channels), len(channels), 2))
    for i, ch in enumerate(channels):
        fXi = p1['f_'+ch]
        if fXi == 0:
            continue
        tXi = p1['t50_'+ch]
        lXi2 = kappa * tXi
        if use_alt:
            lXi2_kappa_alt = kappa_alt * tXi
        sigmaXi2 = sigma_inj[ch]**2
        if beam:
            sigmaXi2 += sigma_beam**2
            
        for j, ch_ in enumerate(channels):
            fYj = p2['f_'+ch_]
            if fYj == 0:
                continue
            tYj = p2['t50_'+ch_]
            lYj2 = kappa * tYj
            if use_alt:
                lYj2_kappa_alt = kappa_alt * tYj
            sigmaYj2 = sigma_inj[ch_]**2
            if beam:
                sigmaYj2 += sigma_beam**2

            # Get alpha, beta, gamma
            alpha[i,j] = ((4 * lcorr**2 - 2 * lXi2 - 2 * lYj2 +
                           sigmaXi2 + sigmaYj2 ) /
                          ( 2 * np.abs(lXi2 - lYj2) +
                            sigmaXi2 + sigmaYj2 )).to('').value
            beta[i,j] = (sigmav * np.abs(tXi - tYj) / 
                         np.sqrt( 4 * lcorr**2 - 2 * lXi2 - 2 * lYj2 +
                                  sigmaXi2 + sigmaYj2 )).to('').value
            gamma[i,j] = (sigmav * np.abs(tXi - tYj) / 
                          np.sqrt( 2 * np.abs( lXi2 - lYj2 ) +
                                   sigmaXi2 + sigmaYj2 )).to('').value
            if use_alt:
                alpha_kappa_alt[i,j] \
                    = ((4 * lcorr**2 - 2 * lXi2 - 2 * lYj2 +
                        sigmaXi2 + sigmaYj2 ) /
                       ( 2 * np.abs(lXi2_kappa_alt - lYj2_kappa_alt) +
                         sigmaXi2 + sigmaYj2 )).to('').value
                alpha_sigmav_alt[i,j] = alpha[i,j]
                alpha_lcorr_alt[i,j] \
                    = ((4 * lcorr_alt**2 - 2 * lXi2 - 2 * lYj2 +
                        sigmaXi2 + sigmaYj2 ) /
                       ( 2 * np.abs(lXi2 - lYj2) +
                         sigmaXi2 + sigmaYj2 )).to('').value
                beta_sigmav_alt[i,j] \
                    = (sigmav_alt * np.abs(tXi - tYj) / 
                       np.sqrt( 4 * lcorr**2 - 2 * lXi2 - 2 * lYj2 +
                                sigmaXi2 + sigmaYj2 )).to('').value
                beta_kappa_alt[i,j] \
                    = (sigmav_alt * np.abs(tXi - tYj) / 
                       np.sqrt( 4 * lcorr**2 - 2 *
                                lXi2_kappa_alt - 2 * lYj2_kappa_alt +
                                sigmaXi2 + sigmaYj2 )).to('').value
                beta_lcorr_alt[i,j] \
                    = (sigmav * np.abs(tXi - tYj) / 
                       np.sqrt( 4 * lcorr_alt**2 -
                                2 * lXi2 - 2 * lYj2 +
                                sigmaXi2 + sigmaYj2 )).to('').value
                gamma_sigmav_alt[i,j] \
                    = (sigmav_alt * np.abs(tXi - tYj) / 
                       np.sqrt( 2 * np.abs( lXi2 - lYj2 ) +
                                sigmaXi2 + sigmaYj2 )).to('').value
                gamma_kappa_alt[i,j] \
                    = (sigmav * np.abs(tXi - tYj) / 
                       np.sqrt( 2 * np.abs( lXi2_kappa_alt - lYj2_kappa_alt ) +
                                sigmaXi2 + sigmaYj2 )).to('').value
                gamma_lcorr_alt[i,j] = gamma[i,j]

            
            # Add contribution
            Xi += fXi * fYj * \
                ( np.log( alpha[i,j] ) +
                  2 * np.sqrt( beta[i,j]**2 / (1 + beta[i,j]**2) ) *
                  np.arcsinh(beta[i,j]) -
                  2 * np.sqrt( gamma[i,j]**2 / (1 + gamma[i,j]**2) ) *
                  np.arcsinh(gamma[i,j]) )
            if use_alt:
                Xi_kappa_alt += fXi * fYj * \
                    ( np.log( alpha_kappa_alt[i,j] ) +
                      2 * np.sqrt( beta_kappa_alt[i,j]**2 /
                                   (1 + beta_kappa_alt[i,j]**2) ) *
                      np.arcsinh(beta_kappa_alt[i,j]) -
                      2 * np.sqrt( gamma_kappa_alt[i,j]**2 /
                                   (1 + gamma_kappa_alt[i,j]**2) ) *
                      np.arcsinh(gamma_kappa_alt[i,j]) )
                Xi_sigmav_alt += fXi * fYj * \
                    ( np.log( alpha_sigmav_alt[i,j] ) +
                      2 * np.sqrt( beta_sigmav_alt[i,j]**2 /
                                   (1 + beta_sigmav_alt[i,j]**2) ) *
                      np.arcsinh(beta_sigmav_alt[i,j]) -
                      2 * np.sqrt( gamma_sigmav_alt[i,j]**2 /
                                   (1 + gamma_sigmav_alt[i,j]**2) ) *
                      np.arcsinh(gamma_sigmav_alt[i,j]) )
                Xi_lcorr_alt += fXi * fYj * \
                    ( np.log( alpha_lcorr_alt[i,j] ) +
                      2 * np.sqrt( beta_lcorr_alt[i,j]**2 /
                                   (1 + beta_lcorr_alt[i,j]**2) ) *
                      np.arcsinh(beta_lcorr_alt[i,j]) -
                      2 * np.sqrt( gamma_lcorr_alt[i,j]**2 /
                                   (1 + gamma_lcorr_alt[i,j]**2) ) *
                      np.arcsinh(gamma_lcorr_alt[i,j]) )

    # Return
    if not return_coef:
        if not use_alt:
            return Xi
        else:
            return Xi, Xi_kappa_alt, Xi_sigmav_alt, Xi_lcorr_alt
    else:
        if not use_alt:
            return Xi, alpha, beta, gamma
        else:
            return Xi, Xi_kappa_alt, Xi_sigmav_alt, Xi_lcorr_alt, \
                alpha, beta, gamma, \
                alpha_kappa_alt, beta_kappa_alt, gamma_kappa_alt, \
                alpha_sigmav_alt, beta_sigmav_alt, gamma_sigmav_alt, \
                alpha_lcorr_alt, beta_lcorr_alt, gamma_lcorr_alt

# Get element-element cross-correlations without normalization factor
for e in prod.keys():

    # Skip elements that have no yield data
    if prod[e]['yld_tot'][-1] == 0:
        continue
    
    # Get mean
    prod[e]['mean'] = lcorr**2 * np.sqrt(Gamma/kappa)
    prod[e]['mean_kappa_alt'] = lcorr**2 * np.sqrt(Gamma/kappa_alt)
    for ch in channels:
        fi = prod[e]['f_'+ch]
        if fi == 0:
            continue
        ti = prod[e]['t50_'+ch]
        l2 = kappa * ti
        l2_kappa_alt = kappa_alt * ti
        prod[e]['mean'] -= fi * l2 * np.sqrt(Gamma/kappa)
        prod[e]['mean_kappa_alt'] -= fi * l2_kappa_alt * \
            np.sqrt(Gamma/kappa_alt)

    # Get correlations
    prod[e]['corr'] = {}
    prod[e]['corr_alt'] = {}
    prod[e]['corr_beam'] = {}
    for e_ in prod.keys():
        if prod[e]['Z'] < Zlo or \
           prod[e]['Z'] > Zhi or \
           e in r_process or \
           prod[e_]['Z'] < Zlo or \
           prod[e_]['Z'] > Zhi or \
           e_ in r_process:
            prod[e]['corr'][e_] = np.nan
        else:
            prod[e]['corr'][e_] = computeXi(prod[e], prod[e_])
            prod[e]['corr_alt'][e_] \
                = np.array(computeXi(prod[e], prod[e_], use_alt=True)[1:])
            prod[e]['corr_beam'][e_] = computeXi(prod[e], prod[e_], beam=True)
            if e == e_:
                # Get dispersion
                prod[e]['sigma2'] = (1 + sigmaw2) / (8 * np.pi) * \
                    prod[e]['corr'][e_]
                prod[e]['sigma2_alt'] = (1 + sigmaw2) / (8 * np.pi) * \
                    prod[e]['corr_alt'][e_]
                prod[e]['sigma2_beam'] = (1 + sigmaw2) / (8 * np.pi) * \
                    prod[e]['corr_beam'][e_]
                
# Normalize and add error effects
for e in prod.keys():
    for e_ in prod.keys():
        if prod[e]['Z'] >= Zlo and \
           prod[e]['Z'] <= Zhi and \
           e not in r_process and \
           prod[e_]['Z'] >= Zlo and \
           prod[e_]['Z'] <= Zhi and \
           e_ not in r_process:
            prod[e]['corr'][e_] *= \
                (1 + sigmaw2) / \
                (8 * np.pi *
                 np.sqrt(prod[e]['sigma2'] * prod[e_]['sigma2']))
            prod[e]['corr_alt'][e_] *= \
                (1 + sigmaw2) / \
                (8 * np.pi *
                 np.sqrt(prod[e]['sigma2_alt'] * prod[e_]['sigma2_alt']))
            prod[e]['corr_beam'][e_] *= \
                (1 + sigmaw2) / \
                (8 * np.pi *
                 np.sqrt(prod[e]['sigma2_beam'] * prod[e_]['sigma2_beam']))
            if e != e_:
                prod[e]['corr_beam'][e_] /= fgas
            
# Functon to plot the correlation matrix
def plot_corr_mat(corr_mat, fignum, Zlo, Zhi, elem_matrix,
                  filename=None, smallfig=False):

    # Initialize figure
    if not smallfig:
        fig = plt.figure(fignum, figsize=(6.5,6.5))
    else:
        fig = plt.figure(fignum, figsize=(4,4))
    fig.clf()
    cbar_min = 0.0
    ax = fig.add_subplot(1,1,1)
    cmap = mpl.colormaps.get_cmap('viridis').copy()
    cmap.set_bad('gray', 1)
    cmap.set_under('white', 1)
    mesh = ax.pcolormesh(np.arange(Zlo, Zhi+2)-0.5,
                         np.arange(Zlo, Zhi+2)-0.5,
                         corr_mat,
                         vmin=cbar_min, vmax=1,
                         cmap=cmap)
    ax.set_xlabel('Z')
    ax.set_ylabel('Z')
    ax.set_xlim(Zlo-0.5, Zhi+0.5)
    ax.set_ylim(Zlo-0.5, Zhi+0.5)

    # Add element names, colored by dominant source
    colors = {
        'snii' : 'C0',
        'snia' : 'C1',
        'agb' : 'C2',
        'ns'  : 'gray'
    }

    # Add striping
    for Z in range(Zlo, Zhi+1):
        e = solar_abd[Z-1]['sym']
        if e in r_process:
            clr = colors['ns']
        else:
            clr = colors[prod[e]['main']]
        ax.fill_between([Z-0.5, Z+0.5], [Z+0.5, Z+0.5],
                        [Zhi+0.5, Zhi+0.5],
                        color=clr,
                        alpha=0.2,
                        lw=0)
        ax.fill_between([Zlo-0.5, Z-0.5], [Z-0.5, Z-0.5],
                        [Z+0.5, Z+0.5],
                        color=clr,
                        alpha=0.2,
                        lw=0)
        
    # y axis
    axty = ax.twinx()
    axty.set_ylim((Zlo-0.5, Zhi+0.5))
    axty.set_yticks(np.arange(Zlo, Zhi+1, 2))
    axty.set_yticklabels(elem_matrix[::2], fontsize=10)
    for t in axty.yaxis.get_ticklabels():
        if t.get_text() in r_process:
            t.set_color(colors['ns'])
        else:
            t.set_color(colors[prod[t.get_text()]['main']])
    axtsy = axty.secondary_yaxis(location='right')
    axtsy.tick_params(axis='y', which='major', pad=21)
    axtsy.set_ylim((Zlo-0.5, Zhi+0.5))
    axtsy.set_yticks(np.arange(Zlo+1, Zhi+1, 2))
    axtsy.set_yticklabels(elem_matrix[1::2], fontsize=10)
    for t in axtsy.yaxis.get_ticklabels():
        if t.get_text() in r_process:
            t.set_color(colors['ns'])
        else:
            t.set_color(colors[prod[t.get_text()]['main']])

    # x axis
    axtx = ax.twiny()
    axtx.set_xlim((Zlo-0.5, Zhi+0.5))
    axtx.set_xticks(np.arange(Zlo, Zhi+1, 2))
    axtx.set_xticklabels(elem_matrix[::2], fontsize=10)
    for t in axtx.xaxis.get_ticklabels():
        if t.get_text() in r_process:
            t.set_color(colors['ns'])
        else:
            t.set_color(colors[prod[t.get_text()]['main']])
    axtsx = axtx.secondary_xaxis(location='top')
    axtsx.tick_params(axis='x', which='major', pad=18)
    axtsx.set_xlim((Zlo-0.5, Zhi+0.5))
    axtsx.set_xticks(np.arange(Zlo+1, Zhi+1, 2))
    axtsx.set_xticklabels(elem_matrix[1::2], fontsize=10)
    for t in axtsx.xaxis.get_ticklabels():
        if t.get_text() in r_process:
            t.set_color(colors['ns'])
        else:
            t.set_color(colors[prod[t.get_text()]['main']])

    # Add inset showing color bar
    axins = inset_axes(ax,
                       width="3%",
                       height="60%",
                       loc="upper left")
    fig.colorbar(mesh, cax=axins, orientation='vertical',
                 label=r'$\Xi_{XY}$')

    # Adjust spacing
    if not smallfig:
        plt.subplots_adjust(left=0.1, bottom=0.1)
    else:
        plt.subplots_adjust(left=0.15, bottom=0.13, right=0.85)

    # Save
    if filename is not None:
        plt.savefig(filename)

# Plot true correlations
elem_matrix = [ s['sym'] for s in solar_abd
                if s['Z'] >= Zlo and s['Z'] <= Zhi ]
ne = len(elem_matrix)
corr_mat = np.zeros((ne,ne))
for i, e in enumerate(elem_matrix):
    for j, e_ in enumerate(elem_matrix):
        if i <= j:
            if e in elem_for_table and e_ in elem_for_table:
                corr_mat[i,j] = prod[e]['corr'][e_]
            else:
                corr_mat[i,j] = np.nan
        else:
            corr_mat[i,j] = -1 # Leave half blank for annotation
plot_corr_mat(corr_mat, 2, Zlo, Zhi, elem_matrix,
              filename=osp.join('output', 'elem_elem_corr.pdf'))

# Save the predicted true cross-correlations to file
for i, e in enumerate(elem_matrix):
    for j, e_ in enumerate(elem_matrix):
        if i >= j:
            corr_mat[i,j] = corr_mat[j,i]
np.savez(osp.join('output', 'cross_corr.npz'),
         elements=np.array(elem_matrix),
         cross_corr=corr_mat)

# Plot observed correlations in gas, only including elements
# observable in gas
for i, e in enumerate(elem_matrix):
    for j, e_ in enumerate(elem_matrix):
        if i <= j:
            if e in elem_for_table and e_ in elem_for_table:
                corr_mat[i,j] = prod[e]['corr_beam'][e_]
            else:
                corr_mat[i,j] = np.nan
        else:
            corr_mat[i,j] = -1 # Leave half blank for annotation
plot_corr_mat(corr_mat[:15,:15], 3, Zlo, 20, elem_matrix[:15],
              smallfig=True,
              filename=osp.join('output', 'elem_elem_corr_gas.pdf'))

# Plot stellar correlations expected for finite errors
for n, s in enumerate(sigma_logZ):
    # Fill correlation matrix, including uncertainty correction
    for i, e in enumerate(elem_matrix):
        for j, e_ in enumerate(elem_matrix):
            if i <= j:
                if e in elem_for_table and e_ in elem_for_table:
                    # Uncertainty factors for X and Y
                    fX = 1 + (prod[e]['mean']**2/prod[e]['sigma2']).\
                        to('').value * \
                        np.log(10)**2 * s**2
                    fY = 1 + (prod[e_]['mean']**2/prod[e_]['sigma2']).\
                        to('').value * \
                        np.log(10)**2 * s**2
                    # Corrected correlation
                    if e != e_:
                        corr_mat[i,j] = prod[e]['corr'][e_] / np.sqrt(fX * fY)
                    else:
                        corr_mat[i,j] = prod[e]['corr'][e_]
                else:
                    corr_mat[i,j] = np.nan
            else:
                corr_mat[i,j] = -1 # Leave half blank for annotation
    plot_corr_mat(corr_mat, 4+n, Zlo, Zhi, elem_matrix,
                  filename=osp.join('output',
                                    'elem_elem_corr_star_{:4.2f}.pdf'.
                                    format(s)))

    
# Read measured correlations from Mead+25
fp = open(osp.join('stellar_data', 'corr_mead25.txt'), 'r')
elem_mead = fp.readline().split()[1:]
fp.close()
corr_mead = np.loadtxt(osp.join('stellar_data', 'corr_mead25.txt'))

# Element by element uncertainties for Mead+25
mead_err = { 'C' : 0.01,
             'O' : 0.013,
             'Na' : 0.014,
             'Mg' : 0.012,
             'Al' : 0.01,
             'Si' : 0.006,
             'S' : 0.018,
             'Ca' : 0.008,
             'Sc' : 0.015,
             'Ti' : 0.008,
             'V' : 0.009,
             'Cr' : 0.008,
             'Mn' : 0.008,
             'Fe' : 0.003,
             'Co' : 0.0085,
             'Ni' : 0.007,
             'Cu' : 0.015,
             'Zn' : 0.014,
             'Sr' : 0.008,
             'Y' : 0.011,
             'Zr' : 0.014,
             'Ba' : 0.011,
             'La' : 0.023,
             'Ce' : 0.018,
             'Pr' : 0.015,
             'Nd' : 0.013,
             'Sm' : 0.019,
             'Eu' : 0.015,
             'Gd' : 0.009,
             'Dy' : 0.0195 }

# Read measured correlations from Ting+22 w/0.1 dex selection
elem_ting = []
fp = open(osp.join('stellar_data', 'corr_ting22_01.txt'), 'r')
for line in fp:
    if line.strip().startswith("#"):
        continue
    spl = line.split()
    e1 = spl[0]
    e2 = spl[1]
    if e1 not in elem_ting:
        elem_ting.append(e1)
    if e2 not in elem_ting:
        elem_ting.append(e2)
fp.seek(0)
corr_ting_01 = np.zeros((len(elem_ting), len(elem_ting)))
for line in fp:
    if line.strip().startswith("#"):
        continue
    spl = line.split()
    e1 = spl[0]
    e2 = spl[1]
    corr_ting_01[elem_ting.index(e1), elem_ting.index(e2)] \
        = float(spl[2])
    corr_ting_01[elem_ting.index(e2), elem_ting.index(e1)] \
        = float(spl[2])
fp.close()

# Read Ting+22 data w/0.05 dex selection
fp = open(osp.join('stellar_data', 'corr_ting22_005.txt'), 'r')
corr_ting_005 = np.zeros((len(elem_ting), len(elem_ting)))
for line in fp:
    if line.strip().startswith("#"):
        continue
    spl = line.split()
    e1 = spl[0]
    e2 = spl[1]
    corr_ting_005[elem_ting.index(e1), elem_ting.index(e2)] \
        = float(spl[2])
    corr_ting_005[elem_ting.index(e2), elem_ting.index(e1)] \
        = float(spl[2])
fp.close()

# Element-by-element errors in the Ting+22 APOGEE sample, in units of dex
apogee_err = { 'O' : 9.58e-3,
               'Ni' : 9.72e-3,
               'Ca' : 1.14e-2,
               'Mn' : 1.26e-2,
               'Si' : 1.43e-2,
               'Al' : 1.80e-2,
               'Co' : 3.08e-2,
               'Cr' : 2.95e-2,
               'K'  : 2.98e-2,
               'Cu' : 3.01e-2,
               'S'  : 3.10e-2,
               'V'  : 3.73e-2,
               'Na' : 3.60e-2,
               'Mg' : 1.07e-2,
               'Fe' : 8.00e-3 }

# Get elements in both our table and observational data
mead_overlap_list = [e for e in elem_for_table if e in elem_mead]
ting_overlap_list = [e for e in elem_for_table if e in elem_ting]

# For each element in each observatonal survey, get the uncertainty
# correction factor
phiX_mead = { e :
              1 + (prod[e]['mean']**2/prod[e]['sigma2']).to('').value *
              np.log(10)**2 * mead_err[e]**2
              for e in mead_overlap_list }
phiX_ting = { e :
              1 + (prod[e]['mean']**2/prod[e]['sigma2']).to('').value *
              np.log(10)**2 * apogee_err[e]**2
              for e in ting_overlap_list }

# Write uncertainties in latex format for convenience
combined_list = list(set(mead_overlap_list + ting_overlap_list))
combined_list.sort(key = lambda e: prod[e]['Z'])
fp = open(osp.join('output', 'stellar_uncertainties.tex'), 'w')
for e in combined_list:
    fp.write("{:d} ({:s}) & ".format(prod[e]['Z'], e))
    if e in mead_err.keys():
        fp.write("{:5.3f} & ".format(mead_err[e]*100))
    else:
        fp.write("- & ")
    if e in apogee_err.keys():
        fp.write("{:5.3f} \\\\\n".format(apogee_err[e]*100))
    else:
        fp.write("- \\\\\n")
fp.close()

# Plot comparison
colors = {
    'snii' : 'b',
    'snia' : 'r',
    'agb' : 'y'
}
srctext = {
    'snii' : 'CCSN',
    'snia' : 'TNSN',
    'agb'  : 'AGB'
}
fig = plt.figure(6, figsize=(4,7))
fig.clf()

# Plot Mead+ 25 comparison
ax = fig.add_subplot(3,1,1)
corr_obs = []
corr_pred = []
for i, e in enumerate(mead_overlap_list):
    for j, e_ in enumerate(mead_overlap_list):
        if i <= j:
            continue
        lcolor = colors[prod[e]['main']]
        rcolor = colors[prod[e_]['main']]
        corr_obs.append(corr_mead[elem_mead.index(e),
                                  elem_mead.index(e_)])
        corr_pred.append(prod[e]['corr'][e_] /
                         np.sqrt( phiX_mead[e] * phiX_mead[e_] ))
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='left'),
                   c=lcolor, s=80, edgecolor='k')
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='right'),
                   c=rcolor, s=80, edgecolor='k')
ax.plot([0,1], [0,1], 'k--')
ax.text(0, 1.15, 'Mead+ 2025', ha='left', va='top')

# Print correlation
coef = pearsonr(corr_obs, corr_pred)
print("Pearson correlation result for model versus Mead+ 2025:"
      " r = {:f}, p = {:e}".format(coef.statistic, coef.pvalue))

# Make legend
handles = []
labels = []
for k in colors.keys():
    s,=ax.plot([-1], [-1], 'o', mec='k', markersize=8.9,
               color=colors[k])
    labels.append(srctext[k])
    handles.append(s)
ax.legend(handles, labels, loc='lower right')

# Adjust limits and labels
xlim=[-0.05,1.05]
ylim=[-0.2, 1.2]
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\Xi_{XY,\mathrm{obs}}$')

# Plot Ting & Weinberg 0.1 dex 2022 comparison
ax = fig.add_subplot(3,1,2)
corr_obs = []
corr_pred = []
for i, e in enumerate(ting_overlap_list):
    for j, e_ in enumerate(ting_overlap_list):
        if i <= j:
            continue
        lcolor = colors[prod[e]['main']]
        rcolor = colors[prod[e_]['main']]
        corr_obs.append(corr_ting_01[elem_ting.index(e),
                                     elem_ting.index(e_)])
        corr_pred.append(prod[e]['corr'][e_] /
                         np.sqrt( phiX_ting[e] * phiX_ting[e_] ))
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='left'),
                   c=lcolor, s=80, edgecolor='k')
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='right'),
                   c=rcolor, s=80, edgecolor='k')
ax.plot([0,1], [0,1], 'k--')
ax.text(0, 1.15, 'Ting \\& Weinberg 2022\n(0.1 dex)', ha='left', va='top')

# Print correlation
coef = pearsonr(corr_obs, corr_pred)
print("Pearson correlation result for model versus Ting & Weinberg "
      "2022 (0.1 dex selection):"
      " r = {:f}, p = {:e}".format(coef.statistic, coef.pvalue))

# Adjust limits and labels
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\Xi_{XY,\mathrm{obs}}$')

# Plot Ting & Weinberg 0.05 dex 2022 comparison
ax = fig.add_subplot(3,1,3)
corr_obs = []
corr_pred = []
for i, e in enumerate(ting_overlap_list):
    for j, e_ in enumerate(ting_overlap_list):
        if i <= j:
            continue
        lcolor = colors[prod[e]['main']]
        rcolor = colors[prod[e_]['main']]
        corr_obs.append(corr_ting_005[elem_ting.index(e),
                                      elem_ting.index(e_)])
        corr_pred.append(prod[e]['corr'][e_] /
                         np.sqrt( phiX_ting[e] * phiX_ting[e_] ))
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='left'),
                   c=lcolor, s=80, edgecolor='k')
        ax.scatter([corr_pred[-1]], [corr_obs[-1]],
                   marker=mpl.markers.MarkerStyle('o', fillstyle='right'),
                   c=rcolor, s=80, edgecolor='k')
ax.plot([0,1], [0,1], 'k--')
ax.text(0, 1.15, 'Ting \\& Weinberg 2022\n(0.05 dex)', ha='left', va='top')

# Print correlation
coef = pearsonr(corr_obs, corr_pred)
print("Pearson correlation result for model versus Ting & Weinberg "
      "2022 (0.05 dex selection):"
      " r = {:f}, p = {:e}".format(coef.statistic, coef.pvalue))

# Adjust limits and labels
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel(r'$\Xi_{XY,\mathrm{pred}}$')
ax.set_ylabel(r'$\Xi_{XY,\mathrm{obs}}$')

# Adjust spacing
plt.subplots_adjust(left=0.18, bottom=0.1, top=0.95, right=0.95, hspace=0.07)

# Save
plt.savefig(osp.join('output', 'Xi_comp.pdf'))


# Plot alternatives
fig = plt.figure(7, figsize=(7,2))
fig.clf()
for i in range(3):

    # Create panel
    #ax = fig.add_subplot(3,1,i+1)
    ax = fig.add_subplot(1,3,i+1)

    # Gather range in predicted correlation
    x = []
    y1 = []
    y2 = []
    for e in prod.keys():
        if prod[e]['yld_tot'][-1] == 0:
            continue
        if np.isnan(prod[e]['corr']['C']):
            continue
        for e_ in prod[e]['corr_alt'].keys():
            x.append(prod[e]['corr'][e_])
            y1.append(prod[e]['corr_alt'][e_][i,0])
            y2.append(prod[e]['corr_alt'][e_][i,1])
    idx = np.argsort(x)
    x = np.array(x)[idx]
    y1 = np.array(y1)[idx]
    y2 = np.array(y2)[idx]

    # Labels
    if i == 0:
        lab = [r'$\kappa/3$', r'$\kappa\times 3$']
        qty = 'kappa'
    elif i == 1:
        lab = [r'$\sigma_v/3$', r'$\sigma_v\times 3$']
        qty = 'sigma_v'
    else:
        lab = [r'$\ell_\mathrm{corr}/2$', r'$\ell_\mathrm{corr}\times 2$']
        qty = 'l_corr'

    # Print typical variation
    print("{:s}: variation by factor of {:f}".
          format(qty,
                 10.**np.nanmedian(np.abs(np.log10(y2/y1))/2)))
        
    # Plot
    ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
    ax.fill_between(x, y1, y2, alpha=0.5)
    ax.plot(x, y1, 'C1', lw=0.5, label=lab[0])
    ax.plot(x, y2, 'C2', lw=0.5, label=lab[1])

    # Add legend
    ax.legend(loc='upper left', prop={'size' : 10})

    # Adjust limits and labels
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    if i > 0:
        ax.set_yticklabels('')
    else:
        ax.set_ylabel(r'$\Xi_{XY,\mathrm{var}}$')
    ax.set_xlabel(r'$\Xi_{XY,\mathrm{fid}}$')

# Adjust spacing
plt.subplots_adjust(left=0.1, bottom=0.25, top=0.95, right=0.97, wspace=0.07)

# Save
plt.savefig(osp.join('output', 'Xi_var.pdf'))
