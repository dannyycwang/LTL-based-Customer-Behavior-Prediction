"""Seed-42 historical Cosmetics diagnostics, checked against the saved full Z matrix."""
from pathlib import Path
import ast,json,time
from collections import Counter
import numpy as np
import pandas as pd
from diagnose_coveo import vector,validate,sha
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]/'work'

def main():
    out=ROOT/'cosmetics_seed42_diagnostics';out.mkdir(exist_ok=True)
    df=pd.read_csv(WORK/'cosmetics_seed42_test.csv',dtype={'session_id_hash':str})
    basis=json.loads((WORK/'cosmetics_BP_basis_clean.json').read_text())
    rec=basis['formulas'][:10];trees=[ast.literal_eval(r['formula_tree']) for r in rec]
    checks=validate(trees);saved=np.load(WORK/'cosmetics_BP_top10_FIXED.npz',allow_pickle=True)
    assert list(saved['formulas'])==[r['formula_str'] for r in rec]
    lookup={str(s):i for i,s in enumerate(saved['session_id_hash'])};inds=np.array([lookup[s] for s in df.session_id_hash])
    assert np.array_equal(saved['y'][inds],df.purchase.to_numpy())
    traces=[[set(t.split('+')) for t in s.split()] for s in df.prop_BP]
    lengths=np.array([len(t) for t in traces]);assert np.array_equal(lengths,df.n_events)
    t0=time.perf_counter();full=np.stack([vector(t,trees) for t in traces]);prefix=np.stack([vector(t[:10],trees) for t in traces])
    mismatches=int((full!=saved['Z'][inds]).sum())
    last20=np.stack([vector(t[-20:],trees) for t in traces])
    last20_mismatches=int((last20!=saved['Z'][inds]).sum())
    union=np.stack([vector([set().union(*t)],trees) for t in traces])
    union_mismatches=int((union!=saved['Z'][inds]).sum())
    pd.DataFrame([dict(reference_scope='full',bit_mismatches=mismatches),dict(reference_scope='last20',bit_mismatches=last20_mismatches),dict(reference_scope='single_state_union_of_full_trace',bit_mismatches=union_mismatches)]).to_csv(out/'saved_matrix_scope_check.csv',index=False)
    if union_mismatches:raise ValueError('Historical parser behavior not reproduced; investigate before downstream training')
    short=lengths<=10;assert np.array_equal(full[short],prefix[short]);delta=prefix!=full;rows=[]
    for j,r in enumerate(rec):
        for group,mask in [('L<=10',short),('L>10',~short),('all',np.ones(len(df),bool))]:
            rows.append(dict(formula_id=r['formula_id'],formula=r['formula_str'],stratum=group,n=int(mask.sum()),changed_n=int(delta[mask,j].sum()),changed_rate=float(delta[mask,j].mean()),zero_to_one_rate=float(((prefix[mask,j]==0)&(full[mask,j]==1)).mean()),one_to_zero_rate=float(((prefix[mask,j]==1)&(full[mask,j]==0)).mean())))
    pd.DataFrame(rows).to_csv(out/'prefix_full_formula_changes.csv',index=False)
    seeds=list(range(202611100,202611110));rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);changed=np.empty_like(full,dtype=bool)
        for i,t in enumerate(traces):
            shuffled=[t[k] for k in rng.permutation(len(t))]
            assert Counter(frozenset(s) for s in shuffled)==Counter(frozenset(s) for s in t)
            changed[i]=vector(shuffled,trees)!=full[i]
        for j,r in enumerate(rec):rows.append(dict(shuffle_seed=seed,formula_id=r['formula_id'],n=len(df),changed_rate=float(changed[:,j].mean())))
        rows.append(dict(shuffle_seed=seed,formula_id='ANY_BIT',n=len(df),changed_rate=float(changed.any(axis=1).mean())))
    pd.DataFrame(rows).to_csv(out/'shuffle_changes.csv',index=False)
    rows=[]
    for group,mask in [('L<=10',short),('L>10',~short)]:rows.append(dict(stratum=group,n=int(mask.sum()),n_positive=int(df.purchase.to_numpy()[mask].sum()),positive_rate=float(df.purchase.to_numpy()[mask].mean())))
    pd.DataFrame(rows).to_csv(out/'length_strata.csv',index=False)
    np.savez_compressed(out/'paired_satisfaction.npz',session_id=df.session_id_hash.to_numpy(dtype=str),label=df.purchase.to_numpy(),length=lengths,prefix=prefix,full=full)
    metadata=dict(status='historical seed-42 Cosmetics test split; newly computed true-full diagnostic features; saved historical Z evaluates a single-state union because of a parser bug; no downstream retraining; cross-seed isolation not yet audited',test_n=len(df),long_n=int((~short).sum()),long_any_changed_n=int(delta[~short].any(axis=1).sum()),long_any_changed_rate=float(delta[~short].any(axis=1).mean()),saved_matrix_vs_true_full_bit_mismatches=mismatches,saved_matrix_vs_last20_bit_mismatches=last20_mismatches,saved_matrix_vs_single_state_union_bit_mismatches=union_mismatches,reference_semantic_checks=checks,shuffle_seeds=seeds,count_feature_dim=len(basis['vocab'])+1,test_csv_sha256=sha(WORK/'cosmetics_seed42_test.csv'),basis_sha256=sha(WORK/'cosmetics_BP_basis_clean.json'),z_sha256=sha(WORK/'cosmetics_BP_top10_FIXED.npz'),elapsed_sec=time.perf_counter()-t0)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
