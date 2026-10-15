import re, os, numpy as np
src=open('/home/claude/titanic_val/ktz_unified_validation.py').read()
blocks=['#'+b for b in re.split(r'(?m)^# %%', src)]
keep=[b for b in blocks if b.startswith('# Block 1:') or b.startswith('# Block 2:') or b.startswith('# Block 9:') or b.startswith('# Block 10:')]
b11=[b for b in blocks if b.startswith('# Block 11:')][0]
b11=b11[:b11.index("cache = OUT_DIR")]
g={}
code='\n'.join(keep)+'\n'+b11
# block 6 helpers may be needed
b6=[b for b in blocks if b.startswith('# Block 6:')][0]
try:
    exec('\n'.join(keep[:2])+'\n'+b6, g)
except Exception as e:
    print('b6 err', e)
exec('\n'.join(keep[2:])+'\n'+b11, g)
globals().update(g)
# exact axisymmetric mean via FFT of the source density lambda0(R) on the grid
src_dens = kt25_Gamma*np.exp(-cR/h_R)*((cR>=cross_source_R_in)&(cR<=cross_source_R_out))
S = np.zeros((NFFT,NFFT)); S[:NX,:NX]=src_dens
FS = np.fft.rfft2(S)
taus = np.linspace(0, kt25_t_corr, 400)
mu_exact = np.zeros((len(cross_elements),NX,NX))
for e in range(len(cross_elements)):
    F = np.zeros_like(FS)
    for c in range(3):
        d=cdelay[e,c]
        if d>=kt25_t_corr or cmean[e,c]==0: continue
        a = cross_sigma_obs[c]**2 + (kt25_sigma_v*d)**2; D = kt25_t_corr-d
        K2s = np.where(K2>0, K2, 1e-30)
        kern = np.exp(-0.5*K2*a)*np.where(K2>0, -np.expm1(-K2s*kt25_kappa*D)/(K2s*kt25_kappa), D)
        F += cmean[e,c]*kt25_event_mass_mean*kern
    # S already density per area; events uniformly in age over t_corr with rate Gamma per unit time
    mu_exact[e]=np.fft.irfft2(FS*F, s=(NFFT,NFFT))[:NX,:NX]
dm = mu_exact - cross_bg
v = dm[:, c_ana]
print('mean fractional error rms per element:', np.round(np.sqrt((v**2).mean(1))/cross_bg[:,c_ana].mean(1),4))
Cerr = v@v.T/v.shape[1] - np.outer(v.mean(1),v.mean(1))*0
# predicted covariance in sim units: need scale of theory comps -> compare to sim stack
Sst=np.load('run/outputs/cross_covariance_stack.npz')['covariance_stack'].mean(0)
R=np.load('run/outputs/cross_results.npz',allow_pickle=True); T=R['Xi_KTZ']
iu=np.triu_indices(16,1)
def xi(C): d=np.sqrt(np.diag(C)); return C/np.outer(d,d)
print('sim bias', (xi(Sst)-T)[iu].mean())
print('sim with mean error removed bias', (xi(Sst-Cerr)-T)[iu].mean(), 'rmse', np.sqrt(((xi(Sst-Cerr)-T)[iu]**2).mean()))
print('frac of diag var from mean err', np.round(np.diag(Cerr)/np.diag(Sst),3))
np.save('chk_mean/Cerr.npy',Cerr)
