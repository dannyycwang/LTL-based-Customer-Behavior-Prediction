from common import *
import shutil

def bank(ds,condition):
    props=json.loads((outdir(ds)/'preparation.json').read_text())['props']
    if condition in ['M1','M3']:return props,json.loads((PREV/ds/'mining/candidates.json').read_text()),'full'
    if condition=='M4':return props+DWELL,json.loads((outdir(ds)/'dwell_candidates.json').read_text()),'dwell'
    return props+DWELL,[dict(formula_tree=t,formula_str=repr(t),family=f,source='DECLARE') for f,t in old.templates(props+DWELL)],'dwell'

def fit(ds):
    dest=outdir(ds);private=privdir(ds)
    if (dest/'linear_freeze.json').exists():return
    data=load(private/'prepared.pkl');del data['test'];tr,va=data['train'],data['val']
    prep=json.loads((dest/'preparation.json').read_text())
    assert sha(PREV/ds/'prepared.pkl')==prep['prior_prepared_sha256']
    previous=load(PREV/ds/'t20/frozen.pkl');pf=json.loads((PREV/ds/'t20/freeze.json').read_text())
    assert sha(PREV/ds/'t20/frozen.pkl')==pf['model_sha256']
    enc=previous['encoder'];ss=previous['summary_scaler'];ts=StandardScaler().fit(tr['time'])
    b0={s:hstack([enc.transform(d['raw']),csr_matrix(ss.transform(d['summary']))],format='csr') for s,d in data.items()}
    base={s:hstack([b0[s],csr_matrix(ts.transform(d['time']))],format='csr') for s,d in data.items()}
    start=time.perf_counter()
    m2=oldfit.model(base['train'],tr['label']);th=threshold(va['label'],m2.predict_proba(base['val'])[:,1])
    timing(ds,'classifier_training',time.perf_counter()-start,condition='M2',boundary='cached input baseline LR fit and validation threshold')
    fitted={'M0':previous['conditions']['A'],'M1':previous['conditions']['C'],'M2':dict(model=m2,threshold=th,selected=[],rows=[])}
    # Verify inherited M0/M1 objective, features, scaler and classifier config.
    assert old.CFG['native_tolerance']==CFG['classifier']['tolerance'] and old.CFG['classifier']['C']==1.
    for s,d in data.items():
        oldd=load(PREV/ds/'prepared.pkl')[20][s]
        for key in ['session_id','label','raw','summary']:np.testing.assert_array_equal(d[key],oldd[key])
    for c in ['M3','M4','M5']:
        if c=='M4':
            while not (dest/'dwell_candidates.json').exists():
                print(ds,'waiting for the single matched dwell mining run',flush=True)
                time.sleep(30)
        props,rows,scope=bank(ds,c);trees=[freeze(r['formula_tree']) for r in rows]
        tick=time.perf_counter();ft=oldfit.bits(tr[scope],props,trees);fv=oldfit.bits(va[scope],props,trees)
        timing(ds,'feature_checking',time.perf_counter()-tick,condition=c)
        sig=set();raw=enc.transform(tr['raw']).tocsc()
        for j in range(raw.shape[1]):sig.add(raw.indices[raw.indptr[j]:raw.indptr[j+1]].astype(np.int64).tobytes())
        for col in np.column_stack([tr['summary'],tr['time']]).T:
            if np.isin(col,[0,1]).all():sig.add(np.flatnonzero(col).astype(np.int64).tobytes())
        target=private/c;target.mkdir(exist_ok=True)
        tick=time.perf_counter();fitted[c]=oldfit.select(base['train'],base['val'],ft,fv,tr['label'],va['label'],rows,target,sig,m2)
        timing(ds,'selection',time.perf_counter()-tick,condition=c,boundary='includes candidate classifier refits; not additive with those refits')
        public=dest/c;public.mkdir(exist_ok=True)
        for name in ['selection.json','selection_steps.csv','selection_trace.csv','excluded.csv','resume_verification.json']:
            if (target/name).exists():shutil.copyfile(target/name,public/name)
        sel=json.loads((public/'selection.json').read_text());sel['conditioning']='M2: R+S+T';save(public/'selection.json',sel)
        dump(private/f'{c}_selected.pkl',fitted[c])
    bundle=dict(encoder=enc,summary_scaler=ss,time_scaler=ts,conditions=fitted)
    dump(private/'linear_frozen.pkl',bundle)
    save(dest/'linear_freeze.json',dict(frozen=True,config_sha256=sha(HERE/'config.json'),model_sha256=sha(private/'linear_frozen.pkl'),prepared_sha256=sha(private/'prepared.pkl'),previous_frozen_sha256=pf['model_sha256'],M0_M1_reuse='exact IDs/labels/raw/summary, inherited encoder/scaler, same native solver/settings',test_evaluated=False))
    print(ds,'six linear conditions frozen',flush=True)

def inputs(ds,d,bundle,c):
    x=hstack([bundle['encoder'].transform(d['raw']),csr_matrix(bundle['summary_scaler'].transform(d['summary']))],format='csr')
    if c not in ['M0','M1']:x=hstack([x,csr_matrix(bundle['time_scaler'].transform(d['time']))],format='csr')
    f=bundle['conditions'][c]
    if f['selected']:
        props,rows,scope=bank(ds,c)
        # M1 bank may contain full candidate list inherited from C.
        ft=oldfit.bits(d[scope],props,[freeze(r['formula_tree']) for r in f['rows']])
        x=oldfit.augment(x,f['formula_scaler'].transform(ft),f['selected'])
    return x

if __name__=='__main__':
    with threadpool_limits(limits=1):fit(sys.argv[1])
