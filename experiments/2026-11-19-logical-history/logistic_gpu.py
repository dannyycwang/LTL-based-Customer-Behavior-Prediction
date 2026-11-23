"""Float64 GPU optimization of liblinear's C=1 L2 logistic objective.
This is linear logistic regression, including L2-penalized intercept (scaling=1).
"""
import numpy as np
from scipy.special import expit
import torch,time

class LogisticGPU:
    def __init__(self,C=1.,max_iter=2000):self.C=C;self.max_iter=max_iter
    def fit(self,x,y,initial=None):
        start=time.perf_counter();torch.set_num_threads(1)
        device='cuda';x=x.tocsr();n,p=x.shape
        xt=torch.sparse_csr_tensor(torch.as_tensor(x.indptr,device=device),torch.as_tensor(x.indices,device=device),torch.as_tensor(x.data,dtype=torch.float64,device=device),size=x.shape,device=device)
        yt=torch.as_tensor(y,dtype=torch.float64,device=device)
        w=torch.nn.Parameter(torch.zeros(p+1,dtype=torch.float64,device=device))
        if initial is not None:
            assert len(initial.coef_[0])<=p
            with torch.no_grad():
                w[:len(initial.coef_[0])].copy_(torch.as_tensor(initial.coef_[0],device=device))
                w[-1]=float(initial.intercept_[0])
        optimizer=torch.optim.LBFGS([w],lr=1.,max_iter=self.max_iter,max_eval=self.max_iter*2,history_size=20,
            tolerance_grad=1e-6,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad(set_to_none=True)
            z=torch.mv(xt,w[:-1])+w[-1]
            loss=self.C*(torch.nn.functional.softplus(z)-yt*z).sum()+.5*torch.dot(w,w)
            loss.backward();return loss
        optimizer.step(closure);loss=closure();state=optimizer.state[w]
        self.objective_=float(loss.detach().cpu());self.gradient_max_=float(w.grad.abs().max().detach().cpu())
        values=w.detach().cpu().numpy();self.coef_=values[:-1][None,:].copy();self.intercept_=values[-1:].copy()
        self.n_iter_=np.array([state['n_iter']]);self.seconds_=time.perf_counter()-start
        self.converged_=self.n_iter_[0]<self.max_iter and self.gradient_max_<.001
        del optimizer,w,xt,yt
        return self
    def predict_proba(self,x):
        p=expit(x@self.coef_[0]+self.intercept_[0]);return np.column_stack([1-p,p])
