"""Label-free train-only redundancy diagnostics from cached feature matrices."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.cache/'provenance.json').read_text())
    with np.load(a.cache/'train_features.npz',allow_pickle=True) as z:
        t={k:z[k] for k in z.files}
    root=Path('C:/Users/ycw/Documents/Codex/2026-11-10/new-chat')
    dataset=meta['dataset']
    if dataset=='Coveo': props=['b','d','a','r','fp','rp','sp','pc','p2','p3','sc','cc','rc']
    else:
        file=(root/'work/electronics_rebuild/seed42_act_rel_features/counts_scaler.npz' if dataset=='Electronics'
              else root/'work/cosmetics_pre_purchase_seed42/features/counts_scaler.npz')
        with np.load(file) as z: props=z['propositions'][:-1].tolist()
    records=meta['candidate_bank']
    for protocol in ['full_observation','next_10']:
        directory=a.results/protocol
        scope='full' if protocol=='full_observation' else 'prefix'
        keep=np.ones(len(t['length']),bool) if scope=='full' else t['length']>=10
        counts=t['counts' if scope=='full' else 'prefix_counts'][keep]
        trans=pd.read_csv(directory/'transitions.csv')
        indices=[props.index(row.from_atom)*len(props)+props.index(row.to_atom) for row in trans.itertuples()]
        base=np.column_stack([counts,t['transition_'+scope][keep][:,indices]])
        _,groups=np.unique(base,axis=0,return_inverse=True)
        f=t['candidate_'+scope][keep]
        df=pd.DataFrame(f)
        minimum=df.groupby(groups).min().to_numpy()
        maximum=df.groupby(groups).max().to_numpy()
        varies=minimum!=maximum
        all_base=np.column_stack([counts,t['transition_'+scope][keep]])
        _,all_groups=np.unique(all_base,axis=0,return_inverse=True)
        all_varies=(df.groupby(all_groups).min().to_numpy()!=df.groupby(all_groups).max().to_numpy())
        selected=json.loads((directory/'selection.json').read_text())['indices']
        x=f.astype(float);x-=x.mean(0)
        norm=np.sqrt((x*x).sum(0))
        denominator=norm[:,None]*norm[selected][None,:]
        corr=np.divide(x.T@x[:,selected],denominator,out=np.zeros_like(denominator),where=denominator>0)
        for k,j in enumerate(selected):corr[j,k]=0
        rows=[]
        for j,r in enumerate(records):
            exact=np.array_equal(f[:,j],counts[:,props.index('a')]>0) if 'a' in props else False
            rows.append(dict(candidate=j,formula=r['formula_str'],
                equal_to_any_cart_on_train=exact,
                max_abs_correlation_with_other_selected=float(np.max(np.abs(corr[j]))),
                identical_B_groups_with_different_formula=int(varies[:,j].sum()),
                identical_counts_ALL_adjacent_groups_with_different_formula=int(all_varies[:,j].sum()),
                total_identical_B_groups=len(minimum),
                training_sessions_in_varying_B_groups=int(varies[groups,j].sum())))
        pd.DataFrame(rows).to_csv(directory/'redundancy.csv',index=False)
        # These witnesses concern Counts + selected Transitions, not all raw data.
        witnesses=[]
        for j in selected:
            eligible=np.flatnonzero(all_varies[:,j])
            if len(eligible):
                idx=np.flatnonzero(all_groups==eligible[0]);zero=idx[f[idx,j]==0][0];one=idx[f[idx,j]==1][0]
                witnesses.append(dict(candidate=j,formula=records[j]['formula_str'],
                    session_false=str(t['session_id'][keep][zero]),session_true=str(t['session_id'][keep][one]),
                    conditioning='identical proposition counts and ALL adjacent-transition occurrence bits',
                    scope=scope))
        (directory/'temporal_witnesses.json').write_text(json.dumps(witnesses,indent=2))
        print(dataset,protocol,'redundancy checks complete',flush=True)

if __name__=='__main__': main()
