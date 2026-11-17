"""Read-only reuse of audited sources; isolated prefix-only derived features."""
from common import *
import time, itertools
from run_submission_controls import audited_inputs,SPLITS
from run_submission_iterations import transition_selection

def templates(props):
    out=[]
    for a in props:out.append(('existence',('F',a)))
    for a,b in itertools.permutations(props,2):
        out.extend([('response',('G',('->',a,('F',b)))),
                    ('precedence',('|',('G',('!',b)),('U',('!',b),('&',a,('!',b))))),
                    ('chain_response',('G',('->',a,('X',b))))])
    return out

def template_bits(trace,props):
    # Same semantics as templates(), O(vocabulary^2 * trace length).
    x=np.array([[a in s for a in props] for s in trace],bool)
    eventual=np.logical_or.accumulate(x[::-1],axis=0)[::-1]
    out=list(x.any(axis=0))
    for a,b in itertools.permutations(range(len(props)),2):
        first_b=np.flatnonzero(x[:,b]); first_a=np.flatnonzero(x[:,a])
        out.extend([np.all(~x[:,a]|eventual[:,b]),
                    not len(first_b) or (bool(len(first_a)) and first_a[0]<first_b[0]),
                    np.all(~x[:-1,a]|x[1:,b]) and not x[-1,a]])
    return np.array(out,dtype=np.uint8)

def check():
    cases=[([{'a'}],('X','true'),0),([{'a'}],('U','a','b'),0),
           ([{'a'},{'b'}],('U','a','b'),1),([{'a','b'}],('&','a','b'),1),
           ([{'a'}],('G',('->','a',('X','b'))),0),([set()],('G',('->','a',('F','b'))),1)]
    for tr,t,w in cases:assert vector(tr,[t])[0]==w
    props=['a','b'];ts=[t for _,t in templates(props)];n=0
    for length in range(1,5):
        for seq in itertools.product([set(),{'a'},{'b'},{'a','b'}],repeat=length):
            assert np.array_equal(template_bits(seq,props),vector(seq,ts));n+=1
    save_json(HERE/'semantic_checks.json',{'hand_cases':len(cases),'exhaustive_template_traces':n,'status':'passed','empty_trace':'rejected; cohort excludes empty histories'})

def batch_template_bits(traces,props):
    x=np.array([[[a in s for a in props] for s in tr] for tr in traces],bool)
    eventual=np.logical_or.accumulate(x[:,::-1,:],axis=1)[:,::-1,:]
    present=x.any(axis=1);first=np.where(present,x.argmax(axis=1),x.shape[1])
    columns=[present[:,j] for j in range(len(props))]
    for a,b in itertools.permutations(range(len(props)),2):
        columns.extend([np.all(~x[:,:,a]|eventual[:,:,b],axis=1),
                        ~present[:,b]|(first[:,a]<first[:,b]),
                        np.all(~x[:,:-1,a]|x[:,1:,b],axis=1)&~x[:,-1,a]])
    return np.stack(columns,axis=1).astype(np.uint8)

def main(dataset):
    out=HERE/'prepared';out.mkdir(exist_ok=True)
    if (out/f'{dataset}.pkl').exists():return
    start=time.perf_counter()
    frames,cols,pr,tc,pt,basis,props,hashes=audited_inputs(dataset)
    records=read_basis(basis);trees=[tree(r) for r in records[:10]]
    alltrees=[tree(r) for r in records]
    specs=templates(props);data={};audit=[];cost=[]
    for split,frame in frames.items():
        rows=[];raw=[];traces=[];t0=time.perf_counter()
        # Parse and retain original cohort/target, no future values used in features.
        for i,row in enumerate(frame[[*cols,tc]].itertuples(index=False,name=None)):
            tr=pt(row[3]); fields=[pr(v) for v in row[:3]]
            assert all(len(f)==len(tr) for f in fields)
            if len(tr)<10:continue
            rows.append((i,len(tr)));raw.append([str(t) for f in fields for t in f[:10]])
            traces.append(tr[:10])
        ix=np.array([r[0] for r in rows]);length=np.array([r[1] for r in rows]);f=frame.iloc[ix]
        d=dict(raw=np.array(raw,object),length=length,session_id=f.session_id_hash.to_numpy(str),
               eventual_label=f.purchase.to_numpy(int),label=((f.purchase.to_numpy()==1)&(length<20)).astype(int))
        if 'user_id' in f:d['user_id']=f.user_id.to_numpy(str)
        cost.append(dict(dataset=dataset,split=split,stage='parse_and_cohort',seconds=time.perf_counter()-t0))
        t0=time.perf_counter();d['counts']=np.array([[sum(a in s for s in tr) for a in props] for tr in traces],np.float32)
        cost.append(dict(dataset=dataset,split=split,stage='counts_preparsed',seconds=time.perf_counter()-t0))
        t0=time.perf_counter();d['logic_all']=np.array([vector(tr,alltrees) for tr in traces],np.uint8);d['ltlf']=d['logic_all'][:,:10]
        cost.append(dict(dataset=dataset,split=split,stage='formula_check_preparsed_all_candidates',seconds=time.perf_counter()-t0))
        t0=time.perf_counter();bits=np.zeros((len(traces),len(props)**2),bool);pi={p:i for i,p in enumerate(props)}
        for j,tr in enumerate(traces):
            for left,right in zip(tr,tr[1:]):
                for a in left:
                    for b in right:
                        if a in pi and b in pi:bits[j,pi[a]*len(props)+pi[b]]=True
        d['transition_all']=bits
        cost.append(dict(dataset=dataset,split=split,stage='transitions_preparsed_all_candidates',seconds=time.perf_counter()-t0))
        t0=time.perf_counter();d['declare_all']=batch_template_bits(traces,props)
        for j in range(min(20,len(traces))):assert np.array_equal(d['declare_all'][j],template_bits(traces[j],props))
        cost.append(dict(dataset=dataset,split=split,stage='declare_preparsed_all_candidates',seconds=time.perf_counter()-t0))
        data[split]=d
        if split!='test':
            for j,r in enumerate(records[:10]):
                col=d['ltlf'][:,j]
                audit.append(dict(dataset=dataset,split=split,formula_index=j+1,formula=r['formula_str'],
                    n=len(col),constant=bool(np.all(col==col[0])),
                    positive_rate=float(col[d['label']==1].mean()),negative_rate=float(col[d['label']==0].mean()),
                    target='next_10',scope='first_10',duplicate_of=','.join(str(k+1) for k in range(j) if np.array_equal(col,d['ltlf'][:,k]))))
        print(dataset,split,len(f),'prepared',flush=True)
    train=data['train'];selected,ranking=transition_selection({'transition_prefix':train['transition_all'],'label':train['eventual_label']},'transition_prefix',props)
    for d in data.values():d['transitions']=d['transition_all'][:,selected].astype(np.float32)
    meta=dict(dataset=dataset,basis_path=str(basis),basis_sha256=sha(basis),split_hashes=hashes,props=props,formulas=records,
              templates=[dict(family=f,tree=t) for f,t in specs],selected_transitions=selected,transition_ranking=ranking,
              protocol_sha256=sha(HERE/'protocol.json'),seconds=time.perf_counter()-start)
    save_json(out/f'{dataset}_manifest.json',meta)
    pd.DataFrame(audit).to_csv(out/f'{dataset}_formula_support.csv',index=False)
    pd.DataFrame(cost).to_csv(out/f'{dataset}_timing.csv',index=False)
    with (out/f'{dataset}.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
    for s,d in data.items():
        pd.DataFrame({k:d[k] for k in ('session_id','user_id','label','eventual_label','length') if k in d}).to_csv(out/f'{dataset}_{s}_ids.csv',index=False)

if __name__=='__main__':
    check()
    for dataset in sys.argv[1:] or ['Coveo','Electronics','Cosmetics']:main(dataset)
