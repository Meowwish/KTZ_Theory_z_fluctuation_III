exec(open('cm_check.py').read().split("for tag,path")[0])
Gwin = (kt25_Gamma*np.exp(-cR/h_R))[c_ana].mean()
cp_abs = kt25_covariance_matrix(cross_sigma_obs, channel_second_moment=kt25_second_yield_normalized)*kt25_event_mass_second*Gwin/(8*np.pi*kt25_kappa)
cp_abs = cp_abs[np.ix_(cross_idx,cross_idx)]
for tag,path in [('clustered','run/outputs'),('poisson','chk_poisson/outputs')]:
    St=np.load(path+'/cross_covariance_stack.npz')['covariance_stack']
    S=St.mean(0)
    st=S-CM_exact
    print(tag,'diag sim/theory CP0:',np.round(np.diag(st)/np.diag(cp_abs),3))
    print(tag,'per-real spread of diag ratio (Fe):',np.round([ (C-CM_exact)[9,9]/cp_abs[9,9] for C in St],2))
    print(tag,'offdiag ratio Fe-Ba, O-Ba:', st[9,14]/cp_abs[9,14], st[2,14]/cp_abs[2,14])
