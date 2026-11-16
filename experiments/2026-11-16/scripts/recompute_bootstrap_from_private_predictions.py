"""Conditional test-sampling SE, using paired cluster bootstrap replicates.

No models are fitted and no thresholds are changed. Standard errors are the
sample SD of bootstrap estimates, NOT divided by sqrt(number of replicates).
"""
from pathlib import Path
import sys, json, time, hashlib
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--workspace',type=Path,required=True,help='Original audit workspace, containing outputs/research_audit and work/*_bootstrap_clusters.csv')
args=parser.parse_args()
ROOT=args.workspace.resolve()
sys.path.insert(0,str(ROOT/'work/python_packages'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(1)
AUD=ROOT/'outputs/research_audit'
OUT=Path(__file__).resolve().parents[1]/'generated/statistics';OUT.mkdir(parents=True,exist_ok=True)
B=1500
DS=['Coveo','Electronics','Cosmetics']
CN=['full_raw','short_raw','prefix_ltlf','counts','full_ltlf']
PREFIX={'Coveo':'protected_first_v1_seed42','Electronics':'electronics_act_rel_seed42_v1','Cosmetics':'cosmetics_pre_purchase_act_rel_seed42_v1'}
LN=['Short raw','Short + Counts','Short + Counts + LTLf','Short + Counts + transitions','Short + Counts + transitions + LTLf']
def clusters(d, ids):
    if d=='Coveo':return np.arange(len(ids)), 'session'
    f=ROOT/f'work/{d.lower()}_bootstrap_clusters.csv'
    users=pd.read_csv(f,dtype=str).set_index('session_id',verify_integrity=True).loc[ids,'user_id']
    assert users.notna().all()
    return pd.factorize(users)[0], 'user'
def scorer(y,p,t):
    order=np.argsort(-p,kind='stable');pp=p[order];yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(y)-1]
    dec=p>=t;both=y*dec
    def f(w):
        n=w.sum();pos=w@y;pred=w@dec;tp=w@both
        den=float(pos)*float(n-pos)*float(pred)*float(n-pred)
        mcc=(float(n)*tp-float(pos)*pred)/np.sqrt(den) if den>0 else 0.
        wo=w[order];tp=np.cumsum(wo*yy)[ends];samples=np.cumsum(wo)[ends]
        precision=np.divide(tp,samples,out=np.zeros(len(samples)),where=samples>0)
        ap=np.dot(np.diff(tp,prepend=0),precision)/pos if pos else np.nan
        return mcc,ap
    return f
rows=[];contrasts=[]
def evaluate(d,setting,ids,y,prob,threshold,names,expected=None):
    start=time.perf_counter();g,unit=clusters(d,ids);ng=int(g.max())+1
    seed=20261113+int(hashlib.sha256((d+setting).encode()).hexdigest()[:6],16)
    rng=np.random.default_rng(seed)
    funcs=[scorer(y,prob[:,j],threshold[j]) for j in range(len(names))]
    point=np.array([f(np.ones(len(y),dtype=np.int64)) for f in funcs])
    if expected is not None:assert np.allclose(point,expected,atol=2e-7,rtol=0),(d,setting,point,expected)
    values=np.empty((B,len(names),2))
    for b in range(B):
        w=np.bincount(rng.integers(ng,size=ng),minlength=ng)[g]
        values[b]=[f(w) for f in funcs]
    assert np.isfinite(values).all()
    np.savez_compressed(OUT/f'{d.lower()}_{setting}_replicates.npz',values=values,point=point,names=names,metrics=['MCC','AP'],seed=seed)
    se=values.std(axis=0,ddof=1)
    for j,c in enumerate(names):
        for k,met in enumerate(['MCC','AP']):
            q=np.quantile(values[:,j,k],[.025,.975])
            rows.append(dict(dataset=d,setting=setting,condition=c,metric=met,estimate=point[j,k],bootstrap_se=se[j,k],lower_95=q[0],upper_95=q[1],n=len(y),unit=unit,n_units=ng,replicates=B))
    pairs=[('full_ltlf','prefix_ltlf'),('full_ltlf','counts'),('counts','full_raw')] if setting.startswith('neural') else [(LN[2],LN[1]),(LN[3],LN[1]),(LN[2],LN[3]),(LN[4],LN[3])]
    for a,b in pairs:
        if a not in names or b not in names:continue
        ia=names.index(a);ib=names.index(b);v=values[:,ia]-values[:,ib];pt=point[ia]-point[ib]
        for k,met in enumerate(['MCC','AP']):
            q=np.quantile(v[:,k],[.025,.975])
            contrasts.append(dict(dataset=d,setting=setting,first=a,second=b,metric=met,estimate=pt[k],bootstrap_se=v[:,k].std(ddof=1),lower_95=q[0],upper_95=q[1],n=len(y),unit=unit,replicates=B))
    pd.DataFrame(rows).to_csv(OUT/'estimates.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    print(f'{d} / {setting}: {len(names)} models, {len(y):,} cases, {time.perf_counter()-start:.1f}s',flush=True)

def main():
    start=time.perf_counter()
    main=pd.read_csv(AUD/'act_rel_seed42_summary/first_purchase_results.csv')
    labels=['Full raw','Short raw','Short+Prefix-LTLf','Short+Counts','Short+Full-LTLf']
    linear=pd.read_csv(AUD/'submission_iterations_seed42/metrics.csv')
    for d in DS:
        ps=[pd.read_csv(AUD/'runs'/f'{PREFIX[d]}_{c}'/'predictions.csv',dtype={'session_id':str}).set_index('session_id',verify_integrity=True) for c in CN]
        for p in ps:assert p[['label','length']].equals(ps[0][['label','length']])
        ids=ps[0].index;y=ps[0].label.to_numpy(np.int64);prob=np.column_stack([p.probability.to_numpy(np.float64) for p in ps]);t=[p.threshold.iloc[0] for p in ps]
        expected=main[main.dataset==d].set_index('condition').loc[labels,['MCC','AUPRC']].to_numpy()
        evaluate(d,'neural_all',ids,y,prob,t,CN,expected)
        for name,mask in [('neural_long',ps[0].length.to_numpy()>10),('neural_short',ps[0].length.to_numpy()<=10)]:
            evaluate(d,name,ids[mask],y[mask],prob[mask],t,CN)
        for proto in ['full_observation','next_5','next_10','next_20']:
            p=pd.read_csv(AUD/'submission_iterations_seed42'/f'{d.lower()}_{proto}_predictions.csv.gz',dtype={'session_id':str}).set_index('session_id',verify_integrity=True)
            met=linear[(linear.dataset==d)&(linear.protocol==proto)].set_index('condition').loc[LN]
            evaluate(d,proto,p.index,p.label.to_numpy(np.int64),p[LN].to_numpy(),met.threshold.to_list(),LN,met[['MCC','AUPRC']].to_numpy())
    meta=dict(replicates=B,base_seed=20261113,elapsed_sec=time.perf_counter()-start,
              definition='SE is sample SD of bootstrap metric estimates; paired deltas computed within shared replicates. No division by sqrt(B).',
              units={'Coveo':'session','Electronics':'user','Cosmetics':'user'},conditional_on='saved fitted models, thresholds, split, formula basis')
    (OUT/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('All bootstrap SEs verified against saved point estimates.',flush=True)
if __name__=='__main__':main()
