"""Audited A/B/C evaluation with train/dev-only greedy formula selection."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AUDIT = Path('C:/Users/ycw/Documents/Codex/2026-11-10/new-chat/outputs/research_audit')
sys.path.insert(0, str(AUDIT))
from diagnose_coveo import freeze, vector, validate, sha
from run_submission_controls import audited_inputs, counts_and_prefix, verify_test_replay, fit_controls
from run_submission_iterations import transition_selection, task_data
from bootstrap_submission_controls import score_function, cluster_ids
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

warnings.simplefilter('error', ConvergenceWarning)

def parse_tree(value):
    return freeze(ast.literal_eval(value) if isinstance(value, str) else value)

def read_basis(path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    return obj['formulas'] if isinstance(obj, dict) else obj

def bank(dataset, basis, electronics):
    file = electronics if dataset == 'Electronics' else HERE / 'inputs' / f'{dataset.lower()}_candidates.csv'
    candidates = pd.read_csv(file).to_dict('records')
    old = read_basis(basis)[:10]
    # Include the historical comparator in the eligible bank without losing provenance.
    rows, seen = [], {}
    for source, records in [('candidate_export', candidates), ('historical_basis', old)]:
        for row in records:
            tree = parse_tree(row['formula_tree'])
            if tree not in seen:
                seen[tree] = len(rows)
                rows.append(dict(formula_tree=tree, formula_str=row['formula_str'], source=source))
    originals = [seen[parse_tree(r['formula_tree'])] for r in old]
    return rows, originals, file

def fit_prob(train, val, yt):
    model = LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=42)
    model.fit(train, yt)
    return model.predict_proba(val)[:, 1]

def select(base_train, base_val, ftrain, fval, ytrain, yval, rows, out):
    started = time.perf_counter()
    scaler = StandardScaler().fit(base_train)
    bt, bv = scaler.transform(base_train), scaler.transform(base_val)
    fs = StandardScaler().fit(ftrain)
    ft, fv = fs.transform(ftrain), fs.transform(fval)
    signatures=[(np.packbits(ftrain[:,j].astype(bool)).tobytes(),
                 np.packbits(fval[:,j].astype(bool)).tobytes()) for j in range(ft.shape[1])]
    # A duplicate column changes effective L2 regularization without adding
    # information. Keep the first representative using TRAINING values only.
    eligible=[];seen_training=set();excluded=[]
    for j in range(ft.shape[1]):
        reason=None
        if rows[j]['source']!='candidate_export': reason='historical_comparator_only'
        elif np.ptp(ftrain[:,j])==0: reason='constant_on_train'
        elif signatures[j][0] in seen_training: reason='duplicate_training_vector'
        if reason:
            excluded.append(dict(candidate=j,reason=reason))
        else:
            seen_training.add(signatures[j][0]);eligible.append(j)
    pd.DataFrame(excluded).to_csv(out/'excluded_from_selection.csv',index=False)
    if len(eligible)<10:
        raise ValueError(f'Only {len(eligible)} nonconstant training-distinct candidates')
    baseline = log_loss(yval, fit_prob(bt, bv, ytrain), labels=[0,1])
    # Training-only eligible representatives; all candidates remain diagnostic.
    selected, history, current = [], [], baseline
    for step in range(10):
        best = None
        def evaluate(j):
            ids = selected + [j]
            loss = log_loss(yval, fit_prob(np.column_stack([bt, ft[:,ids]]),
                                          np.column_stack([bv, fv[:,ids]]), ytrain), labels=[0,1])
            return j,loss
        remaining=[j for j in eligible if j not in selected]
        representatives={}
        for j in remaining:
            representatives.setdefault(signatures[j],j)
        with ThreadPoolExecutor(max_workers=4) as executor:
            unique_scores=dict(executor.map(evaluate,representatives.values()))
        evaluated=[(j,unique_scores[representatives[signatures[j]]]) for j in remaining]
        for j,loss in evaluated:
            history.append(dict(step=step+1, candidate=j, validation_log_loss=loss,
                                improvement=current-loss, formula=rows[j]['formula_str']))
            if best is None or (loss,j) < best:
                best = (loss,j)
        if best is None:
            raise ValueError('Fewer than ten exported candidates; cannot meet K=10')
        current, j = best
        selected.append(j)
        print('greedy',step+1,j,'loss',current, rows[j]['formula_str'],flush=True)
        pd.DataFrame(history).to_csv(out/'selection_trace.csv',index=False)
    result = dict(indices=selected, baseline_validation_log_loss=baseline,
                  final_validation_log_loss=current, seconds=time.perf_counter()-started,
                  conditioning='Counts + fixed selected Transitions; no raw prefix',
                  metric='validation log loss', K=10, eligible_count=len(eligible),
                  duplicate_policy='first exported representative by exact training signature; constants excluded')
    (out/'selection.json').write_text(json.dumps(result,indent=2))
    return selected

def diagnose(tables, counts, base, rows, out):
    ft, fv = tables['train']['candidate'], tables['val']['candidate']
    yt, yv = tables['train']['label'], tables['val']['label']
    fs = StandardScaler().fit(ft)
    zt,zv=fs.transform(ft),fs.transform(fv)
    signatures=[(np.packbits(ft[:,j].astype(bool)).tobytes(),np.packbits(fv[:,j].astype(bool)).tobytes())
                for j in range(ft.shape[1])]
    representatives={}
    for j,sig in enumerate(signatures): representatives.setdefault(sig,j)
    result=[dict(candidate=j, formula=r['formula_str'], train_frequency=float(ft[:,j].mean()),
                 train_positive_frequency=float(ft[yt==1,j].mean()),
                 train_negative_frequency=float(ft[yt==0,j].mean()),
                 validation_frequency=float(fv[:,j].mean()),
                 constant_on_train=bool(np.ptp(ft[:,j])==0)) for j,r in enumerate(rows)]
    for name, values in [('counts', counts),('counts_transitions',base)]:
        scaler=StandardScaler().fit(values['train'])
        bt,bv=scaler.transform(values['train']),scaler.transform(values['val'])
        loss=log_loss(yv,fit_prob(bt,bv,yt),labels=[0,1])
        def evaluate(j):
            ll=log_loss(yv,fit_prob(np.column_stack([bt,zt[:,j]]),
                                  np.column_stack([bv,zv[:,j]]),yt),labels=[0,1])
            return j,ll
        with ThreadPoolExecutor(max_workers=4) as executor:
            unique_scores=dict(executor.map(evaluate,representatives.values()))
        evaluated=[(j,unique_scores[representatives[signatures[j]]]) for j in range(ft.shape[1])]
        for j,ll in evaluated:
            result[j][name+'_validation_log_loss_gain']=loss-ll
    pd.DataFrame(result).to_csv(out/'candidate_diagnostics.csv',index=False)

def bootstrap(dataset, protocol, tables, metrics, out):
    pred=pd.read_csv(out/f'{dataset.lower()}_{protocol}_predictions.csv.gz',dtype={'session_id':str})
    assert np.array_equal(pred.session_id.to_numpy(),tables['test']['session_id'].astype(str))
    y=pred.label.to_numpy()
    groups,unit=cluster_ids(dataset,pred.session_id)
    funcs={r['condition']:score_function(y,pred[r['condition']].to_numpy(),r['threshold']) for r in metrics}
    rng=np.random.default_rng(20261117)
    n=int(groups.max())+1
    samples={k:[] for k in funcs}
    for _ in range(1500):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[groups]
        for k,fn in funcs.items(): samples[k].append(fn(w))
    arrays={k:np.asarray(v) for k,v in samples.items()}
    estimates={k:fn(np.ones(len(y),dtype=int)) for k,fn in funcs.items()}
    results=[]
    for first,second in [(k,None) for k in funcs]+[('B','A'),('C','A'),('C','B')]+([('C','D'),('D','B')] if 'D' in funcs else []):
        a=arrays[first] if second is None else arrays[first]-arrays[second]
        p=estimates[first] if second is None else estimates[first]-estimates[second]
        for j,metric in enumerate(['MCC','AP']):
            lo,hi=np.nanquantile(a[:,j],[.025,.975])
            results.append(dict(first=first,second=second,metric=metric,estimate=p[j],
                                se=np.nanstd(a[:,j],ddof=1),lower95=lo,upper95=hi,
                                unit=unit,replicates=1500,valid_replicates=int(np.isfinite(a[:,j]).sum())))
    pd.DataFrame(results).to_csv(out/'bootstrap.csv',index=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--electronics',type=Path)
    parser.add_argument('--dataset',choices=['Coveo','Electronics','Cosmetics'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--feature-cache',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    dataset=args.dataset
    args.out.mkdir(parents=True,exist_ok=args.resume)
    started=time.perf_counter()
    frames,cols,raw_parse,trace_col,trace_parse,basis,props,hashes=audited_inputs(dataset)
    rows,originals,candidate_file=bank(dataset,basis,args.electronics)
    trees=[r['formula_tree'] for r in rows]
    semantic_checks=validate(trees)
    if args.feature_cache:
        cached=json.loads((args.feature_cache/'provenance.json').read_text())
        assert cached['split_hashes']==hashes and cached['basis_sha256']==sha(basis)
        assert cached['candidate_sha256']==sha(candidate_file)
        data={}
        for split in frames:
            with np.load(args.feature_cache/f'{split}_features.npz',allow_pickle=True) as a:
                data[split]={k:a[k] for k in a.files}
            assert np.array_equal(data[split]['session_id'].astype(str),frames[split].session_id_hash.astype(str))
        actions=[i for i,p in enumerate(props) if p in {'a','b','d','r'}]
    else:
        data,actions,_=counts_and_prefix(frames,cols,raw_parse,trace_col,trace_parse,basis,props,with_transitions=True)
    verify_test_replay(dataset,data['test'])
    for split,frame in frames.items():
        if args.feature_cache: continue
        full,prefix=[],[]
        for v in frame[trace_col]:
            trace=trace_parse(v)
            full.append(vector(trace,trees));prefix.append(vector(trace[:10],trees))
        data[split]['candidate_full']=np.asarray(full)
        data[split]['candidate_prefix']=np.asarray(prefix)
        assert np.array_equal(data[split]['candidate_full'][:,originals],data[split]['full_formula'])
        assert np.array_equal(data[split]['candidate_prefix'][:,originals],data[split]['prefix_formula'])
        np.savez_compressed(args.out/f'{split}_features.npz',**data[split])
    provenance=dict(dataset=dataset,
        split_hashes=hashes,basis_sha256=sha(basis),candidate_sha256=sha(candidate_file),
        candidate_file=str(candidate_file),original_indices=originals,
        semantic_checks=semantic_checks,feature_seconds=time.perf_counter()-started,
        candidate_bank=rows)
    if args.resume and (args.out/'provenance.json').exists():
        previous=json.loads((args.out/'provenance.json').read_text())
        for key in ['dataset','split_hashes','basis_sha256','candidate_sha256','original_indices']:
            assert previous[key]==provenance[key], f'Resume mismatch: {key}'
    else:
        (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(dataset,'features ready',len(rows),'candidates',flush=True)
    for future in [None,10]:
        mode='full_observation' if future is None else 'next_10'
        scope='full' if future is None else 'prefix'
        out=args.out/mode;out.mkdir(exist_ok=args.resume)
        if args.resume and (out/'bootstrap.csv').exists(): continue
        train=data['train']
        transition_train=train if future is None else {k:v[train['length']>=10] for k,v in train.items()}
        ti,tr=transition_selection(transition_train,'transition_'+scope,props)
        pd.DataFrame(tr).to_csv(out/'transitions.csv',index=False)
        tables=task_data(data,future)
        counts={s:(np.column_stack([t['counts'],t['length']]) if future is None else t['prefix_counts']) for s,t in tables.items()}
        base={s:np.column_stack([counts[s],t['transition_'+scope][:,ti]]) for s,t in tables.items()}
        for t in tables.values(): t['candidate']=t['candidate_'+scope]
        if args.resume and (out/'selection.json').exists():
            saved=json.loads((out/'selection.json').read_text())
            assert saved['duplicate_policy'].startswith('first exported representative')
            selected=saved['indices'];assert len(set(selected))==10
        else:
            selected=select(base['train'],base['val'],tables['train']['candidate'],tables['val']['candidate'],
                            tables['train']['label'],tables['val']['label'],rows,out)
        if not (args.resume and (out/'candidate_diagnostics.csv').exists()):
            diagnose(tables,counts,base,rows,out)
        # Same-pool standalone ranking control isolates changed candidate generation.
        ft=tables['train']['candidate']
        standalone=[]
        if dataset=='Electronics':
            ranked=pd.read_csv(args.electronics.parent/'BP_retrained_standalone_top10.csv')
            tree_index={r['formula_tree']:i for i,r in enumerate(rows)}
            standalone=[tree_index[parse_tree(v)] for v in ranked.formula_tree]
        numerical={}
        for s,t in tables.items():
            numerical[s]={'A':base[s],
                          'B':np.column_stack([base[s],t['candidate'][:,originals]]),
                          'C':np.column_stack([base[s],t['candidate'][:,selected]])}
            if dataset=='Electronics':
                numerical[s]['D']=np.column_stack([base[s],t['candidate'][:,standalone]])
        # All selection is frozen before fit_controls reads test outcomes.
        metrics=fit_controls(dataset,tables,mode,actions,out,numerical=numerical)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        selected_rows=[]
        for condition,indices in [('B',originals),('C',selected)]+([('D',standalone)] if dataset=='Electronics' else []):
            for rank,j in enumerate(indices,1):
                selected_rows.append(dict(condition=condition,rank=rank,candidate=j,**rows[j]))
        pd.DataFrame(selected_rows).to_csv(out/'selected_formulas.csv',index=False)
        np.savetxt(out/'selected_train_correlation.csv',np.corrcoef(ft[:,selected].T),delimiter=',')
        bootstrap(dataset,mode,tables,metrics,out)
        print(dataset,mode,'complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
