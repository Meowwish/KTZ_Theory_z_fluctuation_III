exec(open('cm_check.py').read().split("for tag,path")[0])
import numpy as np
R=np.load('run/outputs/cross_results.npz',allow_pickle=True)
orig=_kt25_kernel
def exact_kernel_factory():
    def k(a,b,c):
        # a=num/den, b=sv dd/sqrt(num), c=sv dd/sqrt(den): reconstruct ln((num+beta)/(den+beta)) = ln((1+b^2)/(1+c^2)) + ln(a)
        return np.log(a)+np.log1p(b**2)-np.log1p(c**2)
    return k
for name,kern in [('KT25 arcsinh',orig),('exact 2D drift',exact_kernel_factory())]:
    g['_kt25_kernel']=kern
    th=ktz_window_components(cross_sigma_obs)
    X=normalize_cross_covariance(sum(th.values()))[np.ix_(cross_idx,cross_idx)]
    d=(R['Xi_sim']-X)[iu]
    print(name,'bias %.4f rmse %.4f max %.4f within2sig %d'%(d.mean(),np.sqrt((d**2).mean()),np.abs(d).max(),(np.abs(d)<=2*R['Xi_sim_err'][iu]).sum()))
    Xp=normalize_cross_covariance(sum(v for k_,v in th.items() if k_!='CNP'))[np.ix_(cross_idx,cross_idx)]
    Sp=np.load('chk_poisson/outputs/cross_covariance_stack.npz')['covariance_stack'].mean(0)
    dp=(normalize_cross_covariance(Sp)-Xp)[iu]; print('   poisson bias %.4f rmse %.4f'%(dp.mean(),np.sqrt((dp**2).mean())))
