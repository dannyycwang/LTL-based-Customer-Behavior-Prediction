"""Recover timestamps, replay ordering, slice FIRST, and generate observable time inputs."""
from common import *
from zipfile import ZipFile

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

def timestamps(dataset,ids):
    cache=privdir(dataset)/'timestamps.pkl'
    if cache.exists():return load(cache)
    meta=json.loads((PREV/dataset/'raw_audit.json').read_text());archive=Path(meta['source'])
    assert sha(archive)==meta['source_sha256']
    parts=[]
    if dataset=='Cosmetics':
        maps=pd.concat([pd.read_csv(old.WORK/f'cosmetics_seed42_{s}.csv',usecols=['user_id','user_session','session_id_hash'],dtype=str) for s in old.SPLITS])
        maps=maps[maps.session_id_hash.isin(ids)];sessions=set(maps.user_session)
        cols=['user_id','user_session','event_time','event_type','price']
    else:cols=['session_id_hash','event_type','product_action','server_timestamp_epoch_ms']
    with ZipFile(archive) as z,z.open(meta['member']) as stream:
        for ch in pd.read_csv(stream,usecols=cols,dtype=str,chunksize=300000):
            ch['row_id']=ch.index
            if dataset=='Coveo':
                ch=ch[ch.session_id_hash.isin(ids)].copy()
                ch['ms']=pd.to_numeric(ch.server_timestamp_epoch_ms,errors='coerce')
                typ=ch.event_type.str.lower().str.replace('_',' ',regex=False)
                ch['action']=ch.product_action.str.lower().map({'detail':'d','add':'a','remove':'r','purchase':'p'}).where(typ=='event product')
                ch.loc[typ.isin(['pageview','page view']),'action']='b'
            else:
                ch=ch[ch.user_session.isin(sessions)].merge(maps,on=['user_id','user_session'],validate='many_to_one')
                ch=ch[pd.to_numeric(ch.price,errors='coerce').fillna(0)>=0].copy()
                dt=pd.to_datetime(ch.event_time.str.replace(' UTC','',regex=False),utc=True,errors='coerce')
                ch['ms']=dt.astype('int64')/1e6;ch.loc[dt.isna(),'ms']=np.nan
                ch['action']=ch.event_type.map({'view':'d','cart':'a','remove_from_cart':'r','purchase':'p'})
            assert ch.ms.notna().all(),'Missing source timestamps: cannot establish same event ordering'
            assert ch.action.notna().all()
            parts.append(ch[['session_id_hash','ms','row_id','action']])
    ev=pd.concat(parts).sort_values(['session_id_hash','ms','row_id'],kind='stable')
    result={sid:(g.ms.to_numpy(),g.action.to_numpy()) for sid,g in ev.groupby('session_id_hash',sort=False)}
    assert set(result)==ids
    dump(cache,result);return result

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

def prepare(ds):
    started=time.perf_counter();dest=outdir(ds);private=privdir(ds)
    if (dest/'preparation.json').exists():return
    all_data=load(PREV/ds/'prepared.pkl');data=all_data[20]
    manifest=json.loads((PREV/ds/'mining/manifest.json').read_text())
    assert sha(PREV/ds/'mining/train.json')==manifest['input_sha256']
    assert sha(PREV/ds/'mining/candidates.json')==manifest['candidates_sha256']
    records=pd.read_csv(PREV/ds/'mining/mining_prefix_ids.csv.gz',dtype={'session_id':str})
    ids=set().union(*(set(d['session_id']) for d in data.values()),set(records.session_id))
    raw=timestamps(ds,ids);props=json.loads((PREV/ds/'preparation.json').read_text())['props']
    counts=[]
    for split,d in data.items():
        gs=[]
        for sid,trace in zip(d['session_id'],d['full']):
            ts,actions=raw[sid];assert len(ts)>=20
            assert all(actions[i] in trace[i] for i in range(20))
            g=gaps(ts,20);gs.append(g)
            mutated=np.r_[ts[:20],np.full(max(0,len(ts)-20),-999999.)]
            np.testing.assert_equal(g,gaps(mutated,20))
        d['gaps']=np.array(gs)
        counts.append(dict(split=split,n=len(gs),known=int(np.isfinite(d['gaps']).sum()),zero=int((d['gaps']==0).sum()),unknown=int(np.isnan(d['gaps']).sum()),terminal_unknown=bool(np.isnan(d['gaps'][:,-1]).all())))
    known=data['train']['gaps'];known=known[np.isfinite(known)]
    cap=float(np.quantile(known,.99));q=np.quantile(np.log1p(np.minimum(known,cap)),[1/3,2/3]).tolist()
    bucket=dict(cap_seconds=cap,q_log=q,fit_split='train',fit_landmark=20,known_gaps=len(known),unknown_not_imputed_for_fitting=True)
    for split,d in data.items():
        feats=[make(tr,g,props,bucket) for tr,g in zip(d['full'],d['gaps'])]
        d['time']=np.stack([f[0] for f in feats]);d['dwell']=[f[1] for f in feats];names=feats[0][2]
        assert np.isfinite(d['time']).all()
        reference=all_data[20][split]
        assert np.array_equal(d['label'],reference['label'])
    oldtrain=json.loads((PREV/ds/'mining/train.json').read_text());mined={0:[],1:[]};original={0:[],1:[]}
    indices={t:{sid:i for i,sid in enumerate(all_data[t]['train']['session_id'])} for t in [10,20,30]}
    for r in records.to_dict('records'):
        sid,t,y=r['session_id'],r['landmark'],r['label'];d=all_data[t]['train'];i=indices[t][sid]
        assert int(d['label'][i])==y
        tr=d['full'][i];ts,act=raw[sid];assert all(act[j] in tr[j] for j in range(t))
        _,extended,_=make(tr,gaps(ts,t),props,bucket)
        original[y].append([sorted(s) for s in tr]);mined[y].append([sorted(s) for s in extended])
    assert original[1]==oldtrain['traces_pos'] and original[0]==oldtrain['traces_neg'] and props==oldtrain['vocab']
    save(private/'mining_train.json',dict(vocab=props+DWELL,traces_pos=mined[1],traces_neg=mined[0],ltlftree=''))
    dump(private/'prepared.pkl',data)
    save(dest/'preparation.json',dict(props=props,dwell_props=DWELL,buckets=bucket,time_dimension=len(names),S_dimension=data['train']['summary'].shape[1],time_names=names,audit=counts,suffix_invariance_checked_all_t20=True,mining_records_verified=True,historical_mining_manifest=manifest,cohort_fingerprint={s:sha(PREV/ds/f'cohort_t20_{s}.csv.gz') for s in old.SPLITS},source_sha256=json.loads((PREV/ds/'raw_audit.json').read_text())['source_sha256'],prior_prepared_sha256=sha(PREV/ds/'prepared.pkl'),prepared_sha256=sha(private/'prepared.pkl'),config_sha256=sha(HERE/'config.json')))
    timing(ds,'preprocessing',time.perf_counter()-started)
    print(ds,'preparation passed',counts,'T dimension',len(names),flush=True)

if __name__=='__main__':
    audit_unit()
    for ds in CFG['datasets']:prepare(ds)
