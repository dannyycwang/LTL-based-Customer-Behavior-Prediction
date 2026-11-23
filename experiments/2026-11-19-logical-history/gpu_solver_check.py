"""Pre-test computational equivalence check, not another research condition."""
from pilot import *
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from logistic_gpu import LogisticGPU
import gc

def main():
    with (HERE/'Cosmetics/prepared.pkl').open('rb') as f:data=pickle.load(f)
    tr=data[10]['train'];va=data[10]['val'];del data;gc.collect()
    enc=OneHotEncoder(handle_unknown='ignore');sc=StandardScaler().fit(tr['summary'])
    xt=hstack([enc.fit_transform(tr['raw']),csr_matrix(sc.transform(tr['summary']))],format='csr')
    xv=hstack([enc.transform(va['raw']),csr_matrix(sc.transform(va['summary']))],format='csr');y=tr['label']
    result={'criterion_fixed_before_check':'converged GPU; relative objective difference <1e-6; max val prediction difference <.002; mean <1e-4; GPU fit >5x faster. No AP/MCC/test data.'}
    start=time.perf_counter();cpu=LogisticRegression(**CFG['classifier']).fit(xt,y);cpu_seconds=time.perf_counter()-start
    objective=.5*(np.sum(cpu.coef_**2)+np.sum(cpu.intercept_**2))+CFG['classifier']['C']*log_loss(y,cpu.predict_proba(xt)[:,1],normalize=False)
    print('CPU reference ready',cpu_seconds,objective,flush=True)
    gpu=LogisticGPU().fit(xt,y);p=cpu.predict_proba(xv)[:,1];q=gpu.predict_proba(xv)[:,1]
    result.update(cpu_seconds=cpu_seconds,gpu_seconds=gpu.seconds_,cpu_objective=objective,gpu_objective=gpu.objective_,gpu_iterations=int(gpu.n_iter_[0]),gpu_gradient_max=gpu.gradient_max_,gpu_converged=gpu.converged_,relative_objective_difference=abs(objective-gpu.objective_)/objective,max_prediction_difference=float(np.max(abs(p-q))),mean_prediction_difference=float(np.mean(abs(p-q))),speed_ratio=cpu_seconds/gpu.seconds_)
    result['passed']=bool(gpu.converged_ and result['relative_objective_difference']<1e-6 and result['max_prediction_difference']<.002 and result['mean_prediction_difference']<1e-4 and result['speed_ratio']>5)
    save(HERE/'gpu_solver_check.json',result);print(result,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
