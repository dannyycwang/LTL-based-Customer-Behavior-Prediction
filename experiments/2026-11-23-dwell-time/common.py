"""Read-only adapters to the frozen logical-history experiment."""
import sys,json,time,pickle,hashlib,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'2026-11-19-logical-history'
sys.path.insert(0,str(PREV))
import pilot as old
import fit as oldfit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import log_loss,average_precision_score,matthews_corrcoef,brier_score_loss
from threadpoolctl import threadpool_limits
# Historical modules prepend their own locations. Restore local module precedence
# so generic names such as report/validate resolve to this experiment.
sys.path.insert(0,str(HERE))
save=old.save;sha=old.sha;freeze=old.freeze
CFG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
PRIVATE=HERE/'private';PRIVATE.mkdir(exist_ok=True)
DWELL=['dl','dm','dh','du']
def load(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(obj,f,protocol=4)
    tmp.replace(path)
def outdir(ds):
    p=HERE/ds;p.mkdir(exist_ok=True);return p
def privdir(ds):
    p=PRIVATE/ds;p.mkdir(exist_ok=True);return p
def threshold(y,p):return float(max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda t:matthews_corrcoef(y,p>=t)))
def timing(ds,stage,seconds,**kw):
    p=outdir(ds)/'runtime.json';r=json.loads(p.read_text()) if p.exists() else []
    r.append(dict(stage=stage,seconds=seconds,**kw));save(p,r)
