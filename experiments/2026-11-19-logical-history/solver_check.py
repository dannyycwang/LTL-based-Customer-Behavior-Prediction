"""Numerical/runtime check of two liblinear algorithms for the SAME LR objective.
No test data, AP, MCC or feature selection. Not a classifier comparison.
"""
from pilot import *
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import warnings,gc

def main():
    with (HERE/'Cosmetics/prepared.pkl').open('rb') as f:all_data=pickle.load(f)
    tr=all_data[10]['train'];va=all_data[10]['val'];del all_data;gc.collect()
    enc=OneHotEncoder(handle_unknown='ignore');sc=StandardScaler().fit(tr['summary'])
    xt=hstack([enc.fit_transform(tr['raw']),csr_matrix(sc.transform(tr['summary']))],format='csr')
    xv=hstack([enc.transform(va['raw']),csr_matrix(sc.transform(va['summary']))],format='csr')
    y=tr['label'];result={};preds={}
    for dual in (False,True):
        name='dual' if dual else 'primal';start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            m=LogisticRegression(**CFG['classifier'],dual=dual).fit(xt,y)
        pt=m.predict_proba(xt)[:,1];preds[name]=m.predict_proba(xv)[:,1]
        obj=.5*(np.sum(m.coef_**2)+np.sum(m.intercept_**2))+CFG['classifier']['C']*log_loss(y,pt,normalize=False)
        result[name]=dict(seconds=time.perf_counter()-start,iterations=m.n_iter_.tolist(),objective=obj,converged=not caught)
        print(name,result[name],flush=True)
    result['max_abs_validation_prediction_difference']=float(np.max(abs(preds['primal']-preds['dual'])))
    result['mean_abs_validation_prediction_difference']=float(np.mean(abs(preds['primal']-preds['dual'])))
    result['relative_objective_difference']=abs(result['primal']['objective']-result['dual']['objective'])/result['primal']['objective']
    result['design_shape']=xt.shape;result['nonzeros']=xt.nnz
    result['criterion_fixed_before_check']='Consider dual only if converged, >2x faster, relative objective difference <1e-6, max prediction difference <.002, mean <1e-4; otherwise retain primal'
    save(HERE/'solver_check.json',result)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
