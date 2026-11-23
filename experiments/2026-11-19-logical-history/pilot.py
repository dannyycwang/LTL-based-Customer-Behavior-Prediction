"""Fixed landmark logical-history pilot. Local private sources remain read-only."""
import os,sys,json,time,pickle,hashlib,argparse,itertools,ast
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
SOURCE=Path(CFG['source_project']); WORK=SOURCE/'work'; AUDIT=SOURCE/'outputs/research_audit'
sys.path[:0]=[str(WORK/'python_packages'),str(AUDIT)]
import numpy as np
import pandas as pd
from zipfile import ZipFile
from collections import Counter
from diagnose_coveo import vector,freeze,sha
from run_submission_controls import audited_inputs
SPLITS=('train','val','test')

def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def abstract(fields,props,dataset):
    ps=Counter();cs=set();prevp=prevc=None;out=[]
    for a,p,c in zip(*fields):
        s={a}; hp=p not in ('P0','UNK','','nan','None');hc=c not in ('C0','UNK','','nan','None')
        if dataset=='Cosmetics':
            s.add('hp' if hp else 'np')
            # Match original Cosmetics vectorized treatment for missing products/categories.
            if prevp is not None:s.add('sp' if p==prevp else 'pc')
            if prevc is not None:s.add('sc' if c==prevc else 'cc')
        if hp:
            if dataset=='Coveo' and prevp is not None:s.add('sp' if p==prevp else 'pc')
            s.add('fp' if ps[p]==0 else 'rp');ps[p]+=1
            if ps[p]>=2:s.add('p2')
            if ps[p]>=3:s.add('p3')
            s.add('dp'+str(min(3,len(ps))));prevp=p
        if hc:
            s.add('hc')
            if dataset=='Coveo' and prevc is not None:s.add('sc' if c==prevc else 'cc')
            if c in cs:s.add('rc')
            cs.add(c);prevc=c
        out.append(s&set(props))
    return out

def summary(trace,props):
    x=np.array([[p in s for p in props] for s in trace],np.float64)
    pairs=x[:-1].T@x[1:]
    rec=[]
    for a in ('b','d','a','r'):
        at=[i for i,s in enumerate(trace) if a in s]
        rec.extend([len(trace)-1-at[-1] if at else len(trace),int(not at)])
    return np.r_[x.sum(0),x.any(0),pairs.ravel()>0,pairs.ravel()/(len(trace)-1),rec]

def templates(props):
    out=[('existence',('F',a)) for a in props]
    for a,b in itertools.permutations(props,2):
        out.extend([('response',('G',('->',a,('F',b)))),('precedence',('|',('G',('!',b)),('U',('!',b),('&',a,('!',b))))),('chain_response',('G',('->',a,('X',b))))])
    return out

def raw_audit(dataset,frames,out):
    """Scan complete source, replay all retained action sequences and purchase positions."""
    dest=out/'raw_audit.pkl'
    if dest.exists():
        recorded=json.loads((out/'raw_audit.json').read_text())
        assert sha(Path(recorded['source']))==recorded['source_sha256'],'Raw source changed since replay cache'
        return pd.read_pickle(dest)
    started=time.perf_counter();allf=pd.concat([f.assign(split=s) for s,f in frames.items()]);ids=set(allf.session_id_hash)
    pieces=[];lo=float('inf');hi=-float('inf')
    if dataset=='Coveo':
        archive=Path(CFG['coveo_raw']);member='train/browsing_train.csv'
        cols=['session_id_hash','event_type','product_action','server_timestamp_epoch_ms']
    else:
        archive=WORK/'cosmetics_dec_source_download.zip';member='2019-Dec.csv'
        cols=['user_id','user_session','event_time','event_type','price']
        maps=pd.read_csv(WORK/'cosmetics_seed42_train.csv',usecols=['user_id','user_session','session_id_hash'],dtype=str)
        maps=pd.concat([maps]+[pd.read_csv(WORK/f'cosmetics_seed42_{s}.csv',usecols=['user_id','user_session','session_id_hash'],dtype=str) for s in ('val','test')])
        maps=maps[maps.session_id_hash.isin(ids)]; sessions=set(maps.user_session)
    with ZipFile(archive) as z,z.open(member) as stream:
        for chunk in pd.read_csv(stream,usecols=cols,dtype=str,chunksize=300000):
            chunk['row_id']=chunk.index
            if dataset=='Coveo':
                stamp=pd.to_numeric(chunk.server_timestamp_epoch_ms,errors='coerce')
                chunk['timestamp']=stamp
                sub=chunk[chunk.session_id_hash.isin(ids)].copy()
                typ=sub.event_type.str.lower().str.replace('_',' ',regex=False)
                sub['action']=sub.product_action.str.lower().map({'detail':'d','add':'a','remove':'r','purchase':'p'}).where(typ=='event product')
                sub.loc[typ.isin(['pageview','page view']),'action']='b'
            else:
                stamp=pd.to_datetime(chunk.event_time.str.replace(' UTC','',regex=False),utc=True,errors='coerce').astype('int64')/1e6
                chunk['timestamp']=stamp
                sub=chunk[chunk.user_session.isin(sessions)].merge(maps,on=['user_id','user_session'],validate='many_to_one')
                sub=sub[pd.to_numeric(sub.price,errors='coerce').fillna(0)>=0].copy()
                sub['action']=sub.event_type.map({'view':'d','cart':'a','remove_from_cart':'r','purchase':'p'})
            lo=min(lo,stamp.min());hi=max(hi,stamp.max())
            pieces.append(sub[['session_id_hash','timestamp','row_id','action']].dropna(subset=['timestamp']))
    ev=pd.concat(pieces).sort_values(['session_id_hash','timestamp','row_id'],kind='stable')
    unknown=ev.action.isna().sum()
    if unknown:raise ValueError(f'{unknown} unrecognized events: cannot silently redefine event horizon')
    ev['position']=ev.groupby('session_id_hash').cumcount()+1
    first=ev[ev.action=='p'].groupby('session_id_hash').position.min()
    ev['first_purchase']=ev.session_id_hash.map(first)
    pre=ev[ev.first_purchase.isna()|(ev.position<ev.first_purchase)]
    result=ev.groupby('session_id_hash').agg(start_ms=('timestamp','min'),end_ms=('timestamp','max'),total_events=('action','size'))
    result['first_purchase']=first;result['pre_actions']=pre.groupby('session_id_hash').action.agg(''.join)
    _,cols,parser,_,_,_,_,_=audited_inputs(dataset)
    ref=allf.set_index('session_id_hash')
    assert set(result.index)==ids
    assert np.array_equal(result.first_purchase.notna().reindex(ref.index),ref.purchase.astype(bool))
    expected=ref[cols[0]].map(lambda s:''.join(parser(s)))
    assert (result.pre_actions.reindex(ref.index).fillna('')==expected).all(),'Raw first-purchase/action replay mismatch'
    gaps=ev.groupby('session_id_hash').timestamp.diff()
    meta=dict(source=str(archive),source_sha256=sha(archive),member=member,source_min_ms=lo,source_max_ms=hi,
              replayed_sessions=len(result),unknown_events=int(unknown),max_within_session_gap_ms=float(gaps.max()),
              seconds=time.perf_counter()-started)
    save(out/'raw_audit.json',meta);result.to_pickle(dest)
    print(dataset,'raw replay passed',meta,flush=True)
    return result

def prepare(dataset):
    out=HERE/dataset;out.mkdir(exist_ok=True);started=time.perf_counter()
    frames,cols,parser,tc,pt,basis,props,hashes=audited_inputs(dataset)
    raw=raw_audit(dataset,frames,out);meta=json.loads((out/'raw_audit.json').read_text())
    cross=set(json.loads((out/'cross_month_sessions.json').read_text())) if dataset=='Cosmetics' else set()
    # No new split: all landmarks inherit the same original membership.
    for a,b in itertools.combinations(SPLITS,2):
        assert not set(frames[a].session_id_hash)&set(frames[b].session_id_hash)
        if dataset=='Cosmetics':assert not set(frames[a].user_id)&set(frames[b].user_id)
    data={t:{} for t in CFG['landmarks']};audit=[];checks=0
    for split,f in frames.items():
        buckets={t:[] for t in data};exclusions={t:Counter() for t in data}
        for row in f.to_dict('records'):
            sid=row['session_id_hash'];r=raw.loc[sid];fields=[parser(row[c]) for c in cols];old=pt(row[tc]);n=len(old)
            assert all(len(v)==n for v in fields)
            for t in data:
                why=None
                if n<t:why='purchase_at_or_before_t' if row['purchase'] else 'ends_before_t'
                # Guard left-truncated sessions before assigning an event index.
                margin=1800000 if dataset=='Coveo' else 86400000
                if dataset=='Coveo': left=r.start_ms<=meta['source_min_ms']+margin
                else:left=False
                if why is None and left:why='left_boundary_uncertain'
                if why is None and sid in cross:why='cross_month_session_excluded'
                y=int(pd.notna(r.first_purchase) and t<r.first_purchase<=t+10)
                resolved=y or r.total_events>=t+10 or pd.notna(r.first_purchase)
                right=dataset=='Coveo' and r.end_ms>meta['source_max_ms']-margin
                if why is None and not resolved and right:why='right_censored_unresolved'
                if why:exclusions[t][why]+=1;continue
                pref=[v[:t] for v in fields];short=[v[-10:] for v in pref]
                fulltr=abstract(pref,props,dataset);shorttr=abstract(short,props,dataset)
                assert fulltr==old[:t],f'Abstract replay {dataset}/{sid}/{t}'
                if t==10:assert shorttr==fulltr
                # Features receive only sliced inputs; altering future suffix cannot enter this API.
                checks+=1
                buckets[t].append(dict(session_id=sid,user_id=row.get('user_id',sid),label=y,
                    raw=[str(v) for field in short for v in field],summary=summary(fulltr,props),
                    full=fulltr,short=shorttr,natural_short_negative=int(not y and r.total_events<t+10),
                    first_purchase=r.first_purchase,total_events=r.total_events))
        for t,rows in buckets.items():
            d={k:np.array([r[k] for r in rows]) for k in ['session_id','user_id','label','raw','summary']}
            d.update(full=[r['full'] for r in rows],short=[r['short'] for r in rows]);data[t][split]=d
            pd.DataFrame([{k:r[k] for k in ['session_id','user_id','label','first_purchase','total_events','natural_short_negative']} for r in rows]).to_csv(out/f'cohort_t{t}_{split}.csv.gz',index=False)
            audit.append(dict(dataset=dataset,split=split,landmark=t,eligible_sessions=len(rows),positive=int(d['label'].sum()),negative=int((d['label']==0).sum()),positive_rate=float(d['label'].mean()),users=len(set(d['user_id'])) if dataset=='Cosmetics' else None,natural_short_negatives=sum(r['natural_short_negative'] for r in rows),**dict(exclusions[t])))
    pd.DataFrame(audit).to_csv(out/'cohort_audit.csv',index=False)
    save(out/'preparation.json',dict(props=props,split_hashes=hashes,checks=checks,seconds=time.perf_counter()-started,config_sha256=sha(HERE/'config.json'),abstract_replay='all eligible prefixes match historical causal abstraction; B reset at window start',boundary=CFG['cosmetics_boundary_policy'] if dataset=='Cosmetics' else '30 minute left/right guard; resolved horizons retained'))
    with (out/'prepared.pkl').open('wb') as handle:pickle.dump(data,handle,protocol=4)
    print(pd.DataFrame(audit).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare']);ap.add_argument('dataset',choices=CFG['datasets']);args=ap.parse_args()
    prepare(args.dataset)
