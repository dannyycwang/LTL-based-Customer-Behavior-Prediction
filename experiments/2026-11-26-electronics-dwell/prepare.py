from common import *
from timing_features import gaps,make,audit_unit
from collections import Counter

def abstract(fields,props):
    seenp=set();seenc=set();out=[]
    for i,(a,p,c) in enumerate(zip(*fields)):
        s={a,'fp' if p not in seenp else 'rp'}
        if i:s|={'sp' if p==fields[1][i-1] else 'pc','sc' if c==fields[2][i-1] else 'cc'}
        if len(seenp|{p})>=2:s.add('p2')
        if len(seenp|{p})>=3:s.add('p3')
        if c in seenc:s.add('rc')
        seenp.add(p);seenc.add(c);out.append(s&set(props))
    return out

def main():
    dest=outdir('Electronics');private=privdir('Electronics');audit_unit();start=time.perf_counter()
    if (dest/'preparation.json').exists():return
    frames,cols,parser,tc,pt,basis,props,hashes=old.audited_inputs('Electronics')
    rebuild=old.WORK/'electronics_rebuild';source=rebuild/'source/events.csv'
    manifest=json.loads((old.AUDIT/'electronics_rebuild/manifest.json').read_text())
    assert sha(source)==manifest['source_csv_sha256']
    ids=set().union(*(set(f.session_id_hash) for f in frames.values()))
    # Reproduce the audited original order, including its tie rule, before slicing.
    df=pd.read_csv(source,dtype={k:'string' for k in ['event_type','product_id','category_id','category_code','brand','user_id','user_session']})
    df['event_time']=pd.to_datetime(df.event_time,utc=True,errors='coerce')
    assert df.event_time.notna().all()
    df=df.dropna(subset=['event_time','event_type','user_session']).copy();df['event_type']=df.event_type.astype(str).str.lower().str.strip();df=df.drop_duplicates()
    low=df.event_time.min().value/1e6;high=df.event_time.max().value/1e6
    df=df.sort_values(['user_session','event_time']);df['session_id_hash']=df.user_session.astype(str);df=df.sort_values(['session_id_hash','event_time'])
    df=df[df.session_id_hash.isin(ids)]
    amap={'view':'d','cart':'a','remove_from_cart':'r','purchase':'p'}
    raw={}
    for sid,g in df.groupby('session_id_hash',sort=False):
        g=g.sort_values('event_time');acts=g.event_type.map(amap).fillna('b').tolist();ts=g.event_time.astype('int64').to_numpy()/1e6
        first=acts.index('p')+1 if 'p' in acts else None
        raw[sid]=(ts,acts,first,g.user_id.astype(str).tolist())
    assert set(raw)==ids
    data={t:{} for t in [10,20,30]};audits=[];checks=0
    for split,f in frames.items():
        rows={t:[] for t in data};excluded={t:Counter() for t in data}
        for r in f.to_dict('records'):
            sid=r['session_id_hash'];ts,acts,first,users=raw[sid];fields=[parser(r[c]) for c in cols];n=len(fields[0]);end=first-1 if first else len(acts)
            assert fields[0]==acts[:end] and bool(first)==bool(r['purchase']), 'Historical order/first-purchase replay mismatch'
            if len(set(users))!=1:
                for t in data:excluded[t]['ambiguous_multiuser_session']+=1
                continue
            assert set(users)=={str(r['user_id'])},'Single-user identity replay mismatch'
            original=pt(r[tc]);assert abstract(fields,props)==original,'Historical Electronics relation replay mismatch'
            for t in data:
                reason='ends_or_purchases_before_landmark' if n<t else None
                if reason is None and ts[0]<=low+CFG['boundary_guard_seconds']*1000:reason='left_boundary_uncertain'
                y=int(first is not None and t<first<=t+10)
                resolved=y or len(acts)>=t+10 or first is not None
                if reason is None and not resolved and ts[-1]>high-CFG['boundary_guard_seconds']*1000:reason='right_censored_unresolved'
                if reason:excluded[t][reason]+=1;continue
                pref=[x[:t] for x in fields];tr=abstract(pref,props);g=gaps(ts,t)
                changed=np.r_[ts[:t],np.full(len(ts)-t,-999999.)];np.testing.assert_equal(g,gaps(changed,t))
                assert tr==original[:t] and np.isnan(g[-1]);checks+=1
                rows[t].append(dict(session_id=sid,user_id=str(r['user_id']),label=y,raw=[str(v) for x in pref for v in x[-10:]],summary=old.summary(tr,props),full=tr,gaps=g,first_purchase=first,total_events=len(acts),natural_short_negative=int(not y and len(acts)<t+10)))
        for t,rs in rows.items():
            assert len(rs)>0
            d={k:np.array([r[k] for r in rs]) for k in ['session_id','user_id','label','raw','summary','gaps']};d['full']=[r['full'] for r in rs];data[t][split]=d
            assert len(np.unique(d['label']))==2, 'Insufficient outcome classes'
            pd.DataFrame([{k:r[k] for k in ['session_id','user_id','label','first_purchase','total_events','natural_short_negative']} for r in rs]).to_csv(private/f'cohort_t{t}_{split}.csv.gz',index=False)
            audits.append(dict(split=split,landmark=t,n=len(rs),positive=int(d['label'].sum()),users=len(set(d['user_id'])),**excluded[t]))
    train=data[20]['train'];known=train['gaps'][np.isfinite(train['gaps'])];cap=float(np.quantile(known,.99));q=np.quantile(np.log1p(np.minimum(known,cap)),[1/3,2/3]).tolist();bucket=dict(cap_seconds=cap,q_log=q,fit_split='train',fit_landmark=20)
    for t,parts in data.items():
        for split,d in parts.items():
            feat=[make(tr,g,props,bucket) for tr,g in zip(d['full'],d['gaps'])];d['time']=np.stack([x[0] for x in feat]);d['dwell']=[x[1] for x in feat];names=feat[0][2]
            assert np.isfinite(d['time']).all()
            for i in range(min(100,len(feat))):
                assert feat[i][1][-1]&set(DWELL)=={'du'}
                assert np.isclose(d['time'][i,20],np.nansum(d['gaps'][i]))
    rng=np.random.RandomState(42);records=[];traces={c:{0:[],1:[]} for c in ['actrel','dwell']};counts=[]
    for t in [10,20,30]:
        d=data[t]['train']
        for y in [1,0]:
            ix=np.flatnonzero(d['label']==y);chosen=rng.choice(ix,min(len(ix),4000//3),replace=False);counts.append(dict(landmark=t,label=y,available=len(ix),sampled=len(chosen)))
            for i in chosen:
                records.append(dict(session_id=d['session_id'][i],landmark=t,label=y))
                for c,key in [('actrel','full'),('dwell','dwell')]:traces[c][y].append([sorted(s) for s in d[key][i]])
    pd.DataFrame(records).to_csv(private/'mining_prefix_ids.csv.gz',index=False)
    for c in traces:save(private/f'{c}_train.json',dict(vocab=props+(DWELL if c=='dwell' else []),traces_pos=traces[c][1],traces_neg=traces[c][0],ltlftree=''))
    for y in [0,1]:assert [[sorted(set(s)-set(DWELL)) for s in tr] for tr in traces['dwell'][y]]==traces['actrel'][y]
    dump(private/'prepared.pkl',data[20]);pd.DataFrame(audits).to_csv(dest/'cohort_audit.csv',index=False)
    overlap={a+'_'+b:len(set(data[20][a]['user_id'])&set(data[20][b]['user_id'])) for a,b in [('train','val'),('train','test'),('val','test')]}
    save(dest/'preparation.json',dict(props=props,dwell_props=DWELL,buckets=bucket,time_names=names,time_dimension=len(names),S_dimension=train['summary'].shape[1],source_sha256=sha(source),split_hashes=hashes,prepared_sha256=sha(private/'prepared.pkl'),cohort_fingerprint={s:sha(private/f'cohort_t20_{s}.csv.gz') for s in old.SPLITS},mining_sampling=counts,mining_records_sha256=sha(private/'mining_prefix_ids.csv.gz'),matching_mining_traces_verified=True,observable_prefix_checks=checks,user_overlap_between_splits=overlap,known_gaps=int(len(known)),zero_gaps=int((known==0).sum()),unknown_gaps=int(np.isnan(train['gaps']).sum()),historical_order_replayed=True,boundary_guard_seconds=1800,semantics='p2/p3 count distinct products, as original Electronics; all timing rebuilt train-only'))
    timing('Electronics','preprocessing',time.perf_counter()-start)
    print(pd.DataFrame(audits).to_string(index=False));print('PREPARATION PASS user overlaps',overlap,flush=True)
if __name__=='__main__':main()
