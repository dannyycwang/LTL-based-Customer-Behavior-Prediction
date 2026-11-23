"""Official LIBLINEAR L2R_LR, same C and penalized bias, with optional init_sol.
Keeps portable NumPy coefficients; native allocations are released after each fit.
"""
import sys,time,ctypes
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor_liblinear'))
from liblinear.liblinearutil import problem,parameter,train

class NativeLogistic:
    def __init__(self,C=1.,tol=1e-4):self.C=C;self.tol=tol
    def fit(self,x,y,initial=None):
        start=time.perf_counter();x=x.tocsr();n,p=x.shape
        prob=problem((2*np.asarray(y)-1).astype(float),x,bias=1.)
        param=parameter(f'-s 0 -c {self.C} -e {self.tol} -B 1 -q')
        assert param.regularize_bias==1
        if initial is not None:
            w0=np.zeros(p+1,np.float64);w0[:initial.coef_.shape[1]]=initial.coef_[0];w0[-1]=initial.intercept_[0]
            param.init_sol=w0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        m=train(prob,param);assert m.get_labels()==[1,-1],m.get_labels()
        weights=np.ctypeslib.as_array(m.w,shape=(p+1,)).copy()
        self.coef_=weights[:-1][None,:];self.intercept_=weights[-1:]
        z=x@weights[:-1]+weights[-1];r=expit(z)-y
        g=np.r_[weights[:-1]+self.C*(x.T@r),weights[-1]+self.C*r.sum()]
        r0=.5-y;g0=np.r_[self.C*(x.T@r0),self.C*r0.sum()]
        self.gradient_norm_=float(np.linalg.norm(g))
        self.gradient_limit_=float(self.tol*max(min(np.sum(y==1),np.sum(y==0)),1)/n*np.linalg.norm(g0))
        self.converged_=self.gradient_norm_<=self.gradient_limit_*(1+1e-6)+1e-8
        self.objective_=float(.5*np.dot(weights,weights)+self.C*np.sum(np.logaddexp(0,z)-y*z))
        self.seconds_=time.perf_counter()-start
        assert self.converged_,(self.gradient_norm_,self.gradient_limit_)
        return self
    def predict_proba(self,x):
        p=expit(x@self.coef_[0]+self.intercept_[0]);return np.column_stack([1-p,p])
