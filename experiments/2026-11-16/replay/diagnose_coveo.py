"""Non-training diagnostics on a named local Coveo candidate dataset.

Does not claim identity with the paper's seed-specific aligned splits.
Fixed basis is the byte-identical local copy of the Drive seed-42 Act+Rel basis.
"""
from __future__ import annotations
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import argparse, ast, hashlib, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = AUDIT_ROOT
DEFAULT_SOURCE=COVEO_DIAGNOSTIC_DATA
DEFAULT_BASIS=COVEO_BASIS
REFERENCE=MODEL_DIR/'ltl_integrated_pytorch_behavior_dwell_automaton_experiment.py'

def freeze(x): return tuple(map(freeze,x)) if isinstance(x,list) else x

def vector(trace,trees):
    """Bit-parallel implementation of strong-next finite LTL, position 0 = low bit."""
    n=len(trace)
    if not n: raise ValueError('Empty trace has no initial position')
    mask=(1<<n)-1
    atoms={}
    for i,state in enumerate(trace):
        for a in state: atoms[a]=atoms.get(a,0)|(1<<i)
    cache={}
    def ev(t):
        if t in cache:return cache[t]
        if isinstance(t,str):
            v=mask if t=='true' else (0 if t=='false' else atoms.get(t,0))
        else:
            op=t[0];a=ev(t[1])
            if op in ('!','not'):v=mask^a
            elif op=='X':v=a>>1
            elif op in ('&','&&','and'):v=a&ev(t[2])
            elif op in ('|','||','or'):v=a|ev(t[2])
            elif op in ('->','=>','implies'):v=(mask^a)|ev(t[2])
            elif op=='U':
                v=ev(t[2]);m=a;s=1
                while s<n:
                    v|=m&(v>>s);m&=m>>s;s*=2
            elif op=='F':
                v=mask if a else 0
                if a:v=(1<<a.bit_length())-1
            elif op=='G':
                neg=mask^a;v=mask^((1<<neg.bit_length())-1) if neg else mask
            else:raise ValueError(f'Unsupported operator: {op}')
        cache[t]=v;return v
    return np.array([ev(t)&1 for t in trees],dtype=np.uint8)

def load_reference(path):
    mod=ast.parse(path.read_text(encoding='utf-8'))
    funcs=[n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='finite_ltl_eval']
    if len(funcs)!=1:raise ValueError('Reference evaluator not uniquely found')
    scope={'Sequence':list}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),scope)
    return scope['finite_ltl_eval']

def validate(trees):
    ref=load_reference(REFERENCE);rng=np.random.default_rng(417)
    tests=trees+['true',('X','true'),('X',('!','a')),('U','a','b'),('!','a'),('F','a'),('G','a')]
    checked=0
    for n in [1,2,3,9,10,11,30,70]:
        for _ in range(24):
            tr=[{a for a in ['a','b','d','rp','sp','p2'] if rng.random()<.35} for __ in range(n)]
            actual=vector(tr,tests)
            expected=np.array([ref(t,tr,0,{}) for t in tests],dtype=np.uint8)
            assert np.array_equal(actual,expected),(n,actual,expected)
            checked+=len(tests)
    assert vector([{'a'}],[('X','true')])[0]==0
    return checked

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--basis',type=Path,default=DEFAULT_BASIS)
    ap.add_argument('--out',type=Path,default=ROOT/'coveo_candidate_diagnostics')
    ap.add_argument('--limit',type=int,default=0,help='0 = all test sessions; positive = smoke subset')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=json.loads(args.basis.read_text(encoding='utf-8'))[:10]
    trees=[freeze(r['formula_tree']) for r in records]
    checks=validate(trees);print('reference semantic comparisons passed:',checks,flush=True)
    cols=['session_id_hash','split','purchase','n_events','action_sequence','product_relation_sequence','category_relation_sequence','dwell_seconds_sequence']
    stats=[];test=[];seen={};duplicates=0;cross=0;bad_len=0;purchase_inputs=0;lengths=[]
    t0=time.perf_counter()
    for df in pd.read_csv(args.source,usecols=cols,chunksize=10000):
        for r in df.itertuples(index=False):
            # Explicit names preserve correctness if CSV column order differs from usecols.
            d=r._asdict(); sid=str(d['session_id_hash']);split=d['split'];y=int(d['purchase'])
            acts=json.loads(d['action_sequence']);L=len(acts)
            duplicates+=sid in seen;cross+=sid in seen and seen[sid]!=split;seen[sid]=split
            bad_len+=L!=int(d['n_events']);purchase_inputs+=any(a in ('p','purchase','5',5) for a in acts)
            dw=json.loads(d['dwell_seconds_sequence'])
            stats.append((split,y,L, float(dw[-1]) if dw else np.nan, float(dw[9]) if len(dw)>10 else np.nan))
            if split=='test' and (not args.limit or len(test)<args.limit):
                pr=json.loads(d['product_relation_sequence']);cr=json.loads(d['category_relation_sequence'])
                if not (L==len(pr)==len(cr)):raise ValueError('Trace field lengths differ')
                tr=[{acts[i]}|set(pr[i])|set(cr[i]) for i in range(L)]
                test.append((sid,y,tr))
    sdf=pd.DataFrame(stats,columns=['split','label','length','last_dwell','boundary_dwell'])
    rows=[]
    for (split,group),g in sdf.assign(length_group=np.where(sdf.length<=10,'L<=10','L>10')).groupby(['split','length_group']):
        rows.append(dict(split=split,length_group=group,n=len(g),n_positive=int(g.label.sum()),positive_rate=g.label.mean(),length_max=g.length.max()))
    pd.DataFrame(rows).to_csv(args.out/'length_strata.csv',index=False)
    sdf.groupby(['split','label']).agg(n=('label','size'),last_dwell_gt0_rate=('last_dwell',lambda s:(s>0).mean()),prefix_boundary_dwell_gt0_rate_among_long=('boundary_dwell',lambda s:(s.dropna()>0).mean())).to_csv(args.out/'dwell_audit.csv')
    ids=[s for s,y,tr in test]; y=np.array([y for s,y,tr in test]);L=np.array([len(tr) for s,y,tr in test]);long=L>10
    prefix=np.stack([vector(tr[:10],trees) for s,y,tr in test]);full=np.stack([vector(tr,trees) for s,y,tr in test])
    assert np.array_equal(prefix[~long],full[~long])
    np.savez_compressed(args.out/'paired_satisfaction.npz',session_id=np.array(ids),label=y,length=L,prefix=prefix,full=full)
    changed=prefix!=full;rows=[]
    for j,r in enumerate(records):
        for label,which in [('all',np.ones(len(L),bool)),('L<=10',~long),('L>10',long)]:
            rows.append(dict(formula_id=r['formula_id'],formula=r['formula_str'],stratum=label,n=int(which.sum()),changed_n=int(changed[which,j].sum()),changed_rate=float(changed[which,j].mean()) if which.any() else np.nan,zero_to_one_rate=float(((prefix[which,j]==0)&(full[which,j]==1)).mean()) if which.any() else np.nan,one_to_zero_rate=float(((prefix[which,j]==1)&(full[which,j]==0)).mean()) if which.any() else np.nan))
    pd.DataFrame(rows).to_csv(args.out/'prefix_full_formula_changes.csv',index=False)
    shuffle_seeds=list(range(202611100,202611110));shuffle_rows=[]
    for seed in shuffle_seeds:
        rng=np.random.default_rng(seed);sh=np.empty_like(full)
        for i,(sid,yi,tr) in enumerate(test):
            perm=rng.permutation(len(tr));shuffled=[tr[k] for k in perm]
            assert Counter(frozenset(s) for s in tr)==Counter(frozenset(s) for s in shuffled)
            sh[i]=vector(shuffled,trees)
        delta=sh!=full
        for j,r in enumerate(records):shuffle_rows.append(dict(shuffle_seed=seed,formula_id=r['formula_id'],n=len(test),changed_n=int(delta[:,j].sum()),changed_rate=float(delta[:,j].mean())))
        shuffle_rows.append(dict(shuffle_seed=seed,formula_id='ANY_BIT',n=len(test),changed_n=int(delta.any(axis=1).sum()),changed_rate=float(delta.any(axis=1).mean())))
        print('shuffle',seed,'any-bit change',delta.any(axis=1).mean(),flush=True)
    pd.DataFrame(shuffle_rows).to_csv(args.out/'shuffle_changes.csv',index=False)
    candidates=[i for i in range(len(test)) if long[i] and changed[i].any()]
    if candidates:
        # No labels or predictions participate in example selection.
        i=min(candidates,key=lambda i:(L[i],ids[i]));j=int(np.flatnonzero(changed[i])[0]);sid,yi,tr=test[i]
        example=dict(selection_rule='Shortest L>10 test trace with any changed bit; ties by session ID; first changed formula in fixed basis order. Labels not used.',session_id=sid,label=int(yi),length=int(L[i]),formula=records[j],prefix_value=int(prefix[i,j]),full_value=int(full[i,j]),trace=[sorted(s) for s in tr])
        (args.out/'example.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    metadata=dict(status='candidate-data diagnostic; paper seed-specific split identity NOT established; no predictor trained',source=str(args.source),source_sha256=sha(args.source),basis=str(args.basis),basis_sha256=sha(args.basis),model_checker_reference=str(REFERENCE),reference_sha256=sha(REFERENCE),semantic_checks=checks,test_n=len(test),test_long_n=int(long.sum()),short_vector_mismatches=int(changed[~long].sum()),long_any_bit_changed_n=int(changed[long].any(axis=1).sum()),long_any_bit_changed_rate=float(changed[long].any(axis=1).mean()),duplicate_session_rows=int(duplicates),cross_split_session_rows=int(cross),user_overlap='unverified: user ID unavailable',bad_n_events=int(bad_len),purchase_in_input_sessions=int(purchase_inputs),shuffle_seeds=shuffle_seeds,elapsed_sec=time.perf_counter()-t0,counts_definition='one count per Act+Rel proposition over the same full symbolic trace, plus length; counts and co-occurrence preserved by symbolic shuffles')
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
