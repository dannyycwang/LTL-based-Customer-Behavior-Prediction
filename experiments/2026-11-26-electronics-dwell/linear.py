from common import *
import shutil

def bank(ds,c):
    props=json.loads((outdir(ds)/'preparation.json').read_text())['props']
    if c in ['M1','M3']:return props,json.loads((outdir(ds)/'actrel_candidates.json').read_text()),'full'
    if c=='M4':return props+DWELL,json.loads((outdir(ds)/'dwell_candidates.json').read_text()),'dwell'
    return props+DWELL,[dict(formula_tree=t,formula_str=repr(t),family=f,source='DECLARE') for f,t in old.templates(props+DWELL)],'dwell'

def signatures(x,d,timing):
    result=set();r=x.tocsc()
    for j in range(r.shape[1]):result.add(r.indices[r.indptr[j]:r.indptr[j+1]].astype(np.int64).tobytes())
    for col in (np.column_stack([d['summary'],d['time']]) if timing else d['summary']).T:
        if np.isin(col,[0,1]).all():result.add(np.flatnonzero(col).astype(np.int64).tobytes())
    return result

def fit(ds='Electronics'):
    dest=outdir(ds);private=privdir(ds)
    if (dest/'linear_freeze.json').exists():return
    data=load(private/'prepared.pkl');del data['test'];tr,va=data['train'],data['val']
    enc=OneHotEncoder(handle_unknown='ignore');enc.fit(tr['raw']);ss=StandardScaler().fit(tr['summary']);ts=StandardScaler().fit(tr['time'])
    raw={s:enc.transform(d['raw']) for s,d in data.items()};b0={s:hstack([raw[s],csr_matrix(ss.transform(d['summary']))],format='csr') for s,d in data.items()};base={s:hstack([b0[s],csr_matrix(ts.transform(d['time']))],format='csr') for s,d in data.items()}
    fitted={}
    for c,b in [('M0',b0),('M2',base)]:
        tick=time.perf_counter();m=oldfit.model(b['train'],tr['label']);th=threshold(va['label'],m.predict_proba(b['val'])[:,1]);fitted[c]=dict(model=m,threshold=th,selected=[],rows=[])
        timing(ds,'classifier_training',time.perf_counter()-tick,condition=c)
    for c in ['M1','M3','M4','M5']:
        props,rows,scope=bank(ds,c);trees=[freeze(r['formula_tree']) for r in rows];tick=time.perf_counter();ft=oldfit.bits(tr[scope],props,trees);fv=oldfit.bits(va[scope],props,trees);timing(ds,'feature_checking',time.perf_counter()-tick,condition=c)
        b=b0 if c=='M1' else base;baseline=fitted['M0' if c=='M1' else 'M2']['model'];target=private/c;target.mkdir(exist_ok=True)
        tick=time.perf_counter();fitted[c]=oldfit.select(b['train'],b['val'],ft,fv,tr['label'],va['label'],rows,target,signatures(raw['train'],tr,c!='M1'),baseline);timing(ds,'selection',time.perf_counter()-tick,condition=c,boundary='Includes candidate refits; not additive with classifier refit times')
        public=dest/c;public.mkdir(exist_ok=True)
        for n in ['selection.json','selection_steps.csv','selection_trace.csv','excluded.csv','resume_verification.json']:
            if (target/n).exists():shutil.copyfile(target/n,public/n)
        dump(private/f'{c}_selected.pkl',fitted[c])
    bundle=dict(encoder=enc,summary_scaler=ss,time_scaler=ts,conditions=fitted);dump(private/'linear_frozen.pkl',bundle)
    save(dest/'linear_freeze.json',dict(frozen=True,config_sha256=sha(HERE/'config.json'),model_sha256=sha(private/'linear_frozen.pkl'),prepared_sha256=sha(private/'prepared.pkl'),M0_M1_reuse=False,test_evaluated=False))
    print('All six Electronics linear conditions frozen',flush=True)

def inputs(ds,d,bundle,c):
    x=hstack([bundle['encoder'].transform(d['raw']),csr_matrix(bundle['summary_scaler'].transform(d['summary']))],format='csr')
    if c not in ['M0','M1']:x=hstack([x,csr_matrix(bundle['time_scaler'].transform(d['time']))],format='csr')
    f=bundle['conditions'][c]
    if f['selected']:
        props,_,scope=bank(ds,c);ft=oldfit.bits(d[scope],props,[freeze(r['formula_tree']) for r in f['rows']]);x=oldfit.augment(x,f['formula_scaler'].transform(ft),f['selected'])
    return x
if __name__=='__main__':
    with threadpool_limits(limits=1):fit()
