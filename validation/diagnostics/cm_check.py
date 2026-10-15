exec(open('mean_check.py').read().split('# exact axisymmetric mean')[0])
import numpy as np
iu=np.triu_indices(16,1)
cTh = np.arctan2(cY, cX)
lam_ax = kt25_Gamma*np.exp(-cR/h_R)*((cR>=cross_source_R_in)&(cR<=cross_source_R_out))
lam_sp = lam_ax*(1+eta*h_func(spiral_phase(np.maximum(cR,1e-3), cTh)))
def smooth(f, var):
    S=np.zeros((NFFT,NFFT)); S[:NX,:NX]=f
    return np.fft.irfft2(np.fft.rfft2(S)*np.exp(-0.5*K2*var), s=(NFFT,NFFT))[:NX,:NX]
dens_sp = (1-cluster_fraction)*lam_sp + cluster_fraction*smooth(lam_sp, cluster_radius**2)
dens_ax = (1-cluster_fraction)*lam_ax + cluster_fraction*smooth(lam_ax, cluster_radius**2)
def mean_field(dens):
    S=np.zeros((NFFT,NFFT)); S[:NX,:NX]=dens; FS=np.fft.rfft2(S)
    out=np.zeros((len(cross_elements),NX,NX)); K2s=np.where(K2>0,K2,1e-30)
    for e in range(len(cross_elements)):
        F=np.zeros_like(FS)
        for c in range(3):
            d=cdelay[e,c]
            if d>=kt25_t_corr or cmean[e,c]==0: continue
            a=cross_sigma_obs[c]**2+(kt25_sigma_v*d)**2; D=kt25_t_corr-d
            F+=cmean[e,c]*kt25_event_mass_mean*np.exp(-0.5*K2*a)*np.where(K2>0,-np.expm1(-K2s*kt25_kappa*D)/(K2s*kt25_kappa),D)
        out[e]=np.fft.irfft2(FS*F,s=(NFFT,NFFT))[:NX,:NX]
    return out
mu_sp=mean_field(dens_sp)
v=(mu_sp-cross_bg)[:,c_ana]
CM_exact=v@v.T/v.shape[1]
for tag,path in [('clustered','run/outputs'),('poisson','chk_poisson/outputs')]:
    S=np.load(path+'/cross_covariance_stack.npz')['covariance_stack'].mean(0)
    th=ktz_window_components(cross_sigma_obs)
    stoch=th['CP0']+(th['CNP'] if tag=='clustered' else 0)
    Xs=normalize_cross_covariance(stoch)[np.ix_(cross_idx,cross_idx)]
    Xsim=normalize_cross_covariance(S-CM_exact)
    print(tag,'stochastic-only: sim-th bias',(Xsim-Xs)[iu].mean(),'rmse',np.sqrt(((Xsim-Xs)[iu]**2).mean()),'max',np.abs(Xsim-Xs)[iu].max())
    print(tag,'CM fraction of diag', np.round(np.diag(CM_exact)/np.diag(S),3))
    # theory CM fraction
    tot=sum(th.values()); print(tag,'theory CM frac', np.round((np.diag(th['CM'])/np.diag(tot))[cross_idx],3))
