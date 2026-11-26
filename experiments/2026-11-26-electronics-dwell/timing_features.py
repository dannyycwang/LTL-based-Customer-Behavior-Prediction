from common import *
def gaps(timestamps,t):
    ts=np.asarray(timestamps,dtype=float)[:t].copy()
    assert len(ts)==t
    g=np.full(t,np.nan);valid=np.isfinite(ts[:-1])&np.isfinite(ts[1:])
    delta=np.diff(ts)/1000.
    if np.any(delta[valid]<0):raise ValueError('negative observable gap')
    g[:-1][valid]=delta[valid]
    return g

def make(trace,g,props,bucket):
    known=np.isfinite(g);v=np.where(known,g,0.)
    q=bucket['q_log'];lv=np.log1p(np.minimum(v,bucket['cap_seconds']))
    tags=np.where(~known,'du',np.where(lv<=q[0],'dl',np.where(lv<=q[1],'dm','dh')))
    ext=[set(s)|{str(tag)} for s,tag in zip(trace,tags)]
    values=list(np.log1p(v[-10:]))+list((~known[-10:]).astype(float))
    names=[f'recent_log_gap_{i}' for i in range(10)]+[f'recent_missing_{i}' for i in range(10)]
    def stats(mask,prefix):
        z=g[mask&known];n=len(z)
        values.extend([float(z.sum()),float(z.mean()) if n else 0.,float(z.std()) if n else 0.,float(z.max()) if n else 0.,n,int(n==0)])
        names.extend([prefix+'_'+k for k in ['sum_sec','mean_sec','std_sec','max_sec','known_n','all_missing']])
    stats(np.ones(len(g),bool),'prefix')
    for tag in DWELL:
        values.extend([np.sum(tags==tag),np.mean(tags==tag)])
        names.extend([tag+'_count',tag+'_frequency'])
    for action in ['b','d','a','r']:stats(np.array([action in s for s in trace]),'action_'+action)
    vocab=props+DWELL;x=np.array([[p in s for p in vocab] for s in ext],float)
    pairs=x[:-1].T@x[1:]
    for i,a in enumerate(vocab):
        for j,b in enumerate(vocab):
            if a in DWELL or b in DWELL:
                values.extend([int(pairs[i,j]>0),pairs[i,j]/(len(g)-1)])
                names.extend([f'pair_{a}_{b}_occurrence',f'pair_{a}_{b}_frequency'])
    return np.asarray(values,float),ext,names

def audit_unit():
    stamp=np.arange(40,dtype=float)*1000;stamp[1]=0
    a=gaps(stamp,20);assert a[0]==0 and np.isnan(a[-1]) and a[1]==2
    changed=stamp.copy();changed[20:]=np.arange(20)*999999
    np.testing.assert_equal(a,gaps(changed,20))
    missing=stamp.copy();missing[4]=np.nan;b=gaps(missing,20)
    assert np.isnan(b[3:5]).all()
    try:gaps([1000,0],2)
    except ValueError:pass
    else:raise AssertionError('negative gap accepted')
    props=['d'];tr=[{'d'} for _ in range(20)];buck=dict(q_log=[0.1,1.],cap_seconds=10)
    x,ext,n=make(tr,a,props,buck);xx,ee,_=make(tr,gaps(changed,20),props,buck)
    np.testing.assert_array_equal(x,xx);assert ext==ee and 'du' in ext[-1] and 'dl' in ext[0]
    save(HERE/'unit_audit.json',dict(passed=True,checks=['suffix timestamp invariance','terminal unknown vs real zero','seconds conversion','negative gap rejection','missing endpoints unknown','feature and proposition alignment']))

