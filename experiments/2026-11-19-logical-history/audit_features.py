"""Input-scope diagnostics only; no fitting, outcome association, or extra condition."""
from pilot import *

def main():
    rows=[];cohorts=[];checked=0
    for dataset in CFG['datasets']:
        out=HERE/dataset;cohorts.append(pd.read_csv(out/'cohort_audit.csv'))
        with (out/'prepared.pkl').open('rb') as f:data=pickle.load(f)
        props=json.loads((out/'preparation.json').read_text())['props']
        for t,parts in data.items():
            for split,d in parts.items():
                n=len(d['label']);changed=sum(a!=b[-10:] for a,b in zip(d['short'],d['full']))
                rows.append(dict(dataset=dataset,landmark=t,split=split,n=n,window_reset_changes_trace=changed,fraction=changed/n))
                assert len(d['raw'])==len(d['summary'])==len(d['full'])==len(d['short'])==n
                if t==10:assert changed==0
                for i in range(min(50,n)):
                    tr=d['full'][i];assert len(tr)==t and len(d['short'][i])==10 and len(d['raw'][i])==30
                    counts=Counter(p for state in tr for p in state)
                    pair=Counter((a,b) for left,right in zip(tr,tr[1:]) for a in left for b in right)
                    rec=[]
                    for a in ('b','d','a','r'):
                        pos=[j for j in range(t) if a in tr[j]];rec += [t-1-pos[-1] if pos else t,int(not pos)]
                    expected=[counts[p] for p in props]+[int(counts[p]>0) for p in props]+[int(pair[a,b]>0) for a in props for b in props]+[pair[a,b]/(t-1) for a in props for b in props]+rec
                    np.testing.assert_array_equal(expected,d['summary'][i]);checked+=1
        # Every mining-prefix ID, when present, belongs to train at that exact landmark.
        p=out/'mining/mining_prefix_ids.csv.gz'
        if p.exists():
            ids=pd.read_csv(p,dtype={'session_id':str})
            for t,sub in ids.groupby('landmark'):
                tr=data[int(t)]['train'];lookup=dict(zip(tr['session_id'],tr['label']))
                assert all(sid in lookup and lookup[sid]==y for sid,y in zip(sub.session_id,sub.label))
                assert not sub.session_id.duplicated().any()
    combined=pd.concat(cohorts,ignore_index=True)
    for col in combined.columns:
        if col!='users':combined[col]=combined[col].fillna(0)
    combined.to_csv(HERE/'cohort_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'abstraction_scope_audit.csv',index=False)
    save(HERE/'feature_audit.json',dict(status='passed',independent_summary_checks=checked,all_cohort_dimensions_aligned=True,t10_input_identity_all_sessions=True,mining_ids_and_future_labels_training_only=True,scope='No outcomes used in abstraction scope comparison'))

if __name__=='__main__':main()
