exec(open('mean_check.py').read().split('# exact axisymmetric mean')[0])
import numpy as np
R=np.load('run/outputs/cross_results.npz',allow_pickle=True)
iu=np.triu_indices(16,1)
def win(sig):
    c=ktz_window_components(sig); X=normalize_cross_covariance(sum(c.values()))
    return X[np.ix_(cross_idx,cross_idx)]
X0=win(kt25_sigma_inj); Xb=win(cross_sigma_obs)
print('stored KTZ vs recomputed', np.abs(Xb-R['Xi_KTZ']).max())
print('beam effect on theory: mean', (Xb-X0)[iu].mean(), 'max', np.abs(Xb-X0)[iu].max())
print('sim - nobeam theory: bias', (R['Xi_sim']-X0)[iu].mean(), 'rmse', np.sqrt(((R['Xi_sim']-X0)[iu]**2).mean()))
print('sigma_inj', kt25_sigma_inj, 'obs', cross_sigma_obs)
