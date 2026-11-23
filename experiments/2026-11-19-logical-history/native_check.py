"""Check official LIBLINEAR cold/warm solves against inherited sklearn primal LR."""
from pilot import *
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from native_logistic import NativeLogistic
from fit import bits
import gc

def compare(cpu,native,xv):
    p=cpu.predict_proba(xv)[:,1];q=native.predict_proba(xv)[:,1]
    return dict(max_prediction_difference=float(np.max(abs(p-q))),mean_prediction_difference=float(np.mean(abs(p-q))),gradient_norm=native.gradient_norm_,gradient_limit=native.gradient_limit_,converged=native.converged_)

def objective(m,x,y):return float(.5*(np.sum(m.coef_**2)+np.sum(m.intercept_**2))+log_loss(y,m.predict_proba(x)[:,1],normalize=False))

def main():
    with (HERE/'Cosmetics/prepared.pkl').open('rb') as f:data=pickle.load(f)
    tr=data[10]['train'];va=data[10]['val'];del data;gc.collect()
    enc=OneHotEncoder(handle_unknown='ignore');sc=StandardScaler().fit(tr['summary'])
    xt=hstack([enc.fit_transform(tr['raw']),csr_matrix(sc.transform(tr['summary']))],format='csr')
    xv=hstack([enc.transform(va['raw']),csr_matrix(sc.transform(va['summary']))],format='csr');y=tr['label']
    result={'objective':'0.5*(||w||^2+b^2)+C*sum logistic loss; C=1; same eps=1e-4 scaled gradient criterion',
        'criterion_fixed_before_check':'Both cold and first eligible candidate warm solves: relative objective difference <1e-6; max val prediction difference <.002; mean <1e-4; analytical gradient convergence passed. Warm candidate >5x faster than inherited cold candidate. No test/AP/MCC.',
        'native_version':'2.50.0','candidate_index':32}
    start=time.perf_counter();cpu=LogisticRegression(**CFG['classifier']).fit(xt,y);sec=time.perf_counter()-start
    native=NativeLogistic(tol=1e-6).fit(xt,y);o=objective(cpu,xt,y)
    result['baseline']=dict(cpu_seconds=sec,native_seconds=native.seconds_,cpu_objective=o,native_objective=native.objective_,relative_objective_difference=abs(o-native.objective_)/o,**compare(cpu,native,xv))
    print('BASELINE',result['baseline'],flush=True)
    props=json.loads((HERE/'Cosmetics/preparation.json').read_text())['props'];pool=json.loads((HERE/'Cosmetics/mining/candidates.json').read_text());tree=freeze(pool[32]['formula_tree'])
    ft=bits(tr['short'],props,[tree]);fv=bits(va['short'],props,[tree]);fs=StandardScaler().fit(ft)
    at=hstack([xt,csr_matrix(fs.transform(ft))],format='csr');av=hstack([xv,csr_matrix(fs.transform(fv))],format='csr')
    start=time.perf_counter();cpuc=LogisticRegression(**CFG['classifier']).fit(at,y);sec=time.perf_counter()-start
    nativec=NativeLogistic(tol=1e-6).fit(at,y,initial=native);o=objective(cpuc,at,y)
    result['candidate']=dict(cpu_seconds=sec,native_seconds=nativec.seconds_,cpu_objective=o,native_objective=nativec.objective_,relative_objective_difference=abs(o-nativec.objective_)/o,speed_ratio=sec/nativec.seconds_,**compare(cpuc,nativec,av))
    result['passed']=bool(all(v['converged'] and v['relative_objective_difference']<1e-6 and v['max_prediction_difference']<.002 and v['mean_prediction_difference']<1e-4 for v in [result['baseline'],result['candidate']]) and result['candidate']['speed_ratio']>5)
    result['native_tolerance']=1e-6
    result['objective']='Same L2 penalized LR objective; native convergence tightened to 1e-6; inherited sklearn tolerance 1e-4'
    save(HERE/'native_check.json',result);print('RESULT',result,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
