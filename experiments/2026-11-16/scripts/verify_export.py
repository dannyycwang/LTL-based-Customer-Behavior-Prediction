"""Verify aggregate evidence, formula semantics, and the snapshot manifest without private data."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'replay'))
from diagnose_coveo import freeze,validate,vector
from fix_cosmetics_parser import parse_event_trace

checked_files=0
manifest=ROOT/'MANIFEST.json'
if manifest.exists():
    for row in json.loads(manifest.read_text(encoding='utf-8'))['files']:
        p=ROOT/row['path']
        assert p.is_file(),row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
        checked_files+=1

semantic_checks=0
for p in sorted((ROOT/'formulas').glob('*.json')):
    obj=json.loads(p.read_text(encoding='utf-8'))
    records=obj if isinstance(obj,list) else obj['formulas']
    trees=[ast.literal_eval(r['formula_tree']) if isinstance(r['formula_tree'],str) else freeze(r['formula_tree']) for r in records[:10]]
    assert len(trees)==10
    semantic_checks+=validate(trees)

toy=[set('B'),set('D'),set('B'),set('D'),set('A'),set('R')]
queries=[('F','A'),('F',('&','D',('X','A'))),('F',('&','A',('X','R')))]
assert vector(toy,queries).tolist()==[1,1,1]
assert vector(toy[:3],queries).tolist()==[0,0,0]
parsed=parse_event_trace('b+fp d+rp a+rp')
assert len(parsed)==3,parsed
assert parsed[1]=={'d','rp'}

est=pd.read_csv(ROOT/'statistics/estimates.csv')
con=pd.read_csv(ROOT/'statistics/contrasts.csv')
count=0
for (ds,setting),frame in est.groupby(['dataset','setting']):
    with np.load(ROOT/'statistics'/f'{ds.lower()}_{setting}_replicates.npz') as a:
        names=list(a['names']);metrics=list(a['metrics'])
        assert a['values'].shape[0]==1500
        for r in frame.itertuples():
            j=names.index(r.condition);k=metrics.index(r.metric)
            assert np.isclose(a['point'][j,k],r.estimate,atol=1e-12,rtol=0)
            assert np.isclose(a['values'][:,j,k].std(ddof=1),r.bootstrap_se,atol=1e-12,rtol=0)
            count+=1
        for r in con[(con.dataset==ds)&(con.setting==setting)].itertuples():
            x=names.index(r.first);y=names.index(r.second);k=metrics.index(r.metric)
            v=a['values'][:,x,k]-a['values'][:,y,k]
            assert np.isclose(v.std(ddof=1),r.bootstrap_se,atol=1e-12,rtol=0)
            assert np.isclose(a['point'][x,k]-a['point'][y,k],r.estimate,atol=1e-12,rtol=0)
            count+=1

# Match the main neural summary to the archived completed runs.
main=pd.read_csv(ROOT/'results/act_rel_seed42_summary/first_purchase_results.csv')
prefix={'Coveo':'protected_first_v1_seed42','Electronics':'electronics_act_rel_seed42_v1','Cosmetics':'cosmetics_pre_purchase_act_rel_seed42_v1'}
conditions={'Full raw':'full_raw','Short raw':'short_raw','Short+Prefix-LTLf':'prefix_ltlf','Short+Counts':'counts','Short+Full-LTLf':'full_ltlf'}
for r in main.itertuples():
    m=json.loads((ROOT/'results/runs'/f'{prefix[r.dataset]}_{conditions[r.condition]}'/'metrics.json').read_text())
    assert np.isclose(m['mcc'],r.MCC,atol=1e-12,rtol=0)
    assert np.isclose(m['auprc'],r.AUPRC,atol=1e-12,rtol=0)
assert len(list((ROOT/'results/runs').glob('*/metrics.json')))==27
print(json.dumps(dict(manifest_files=checked_files,formula_semantic_comparisons=semantic_checks,
    bootstrap_estimates_and_contrasts=count,completed_neural_runs=27,
    main_neural_rows=len(main),private_data_required=False),indent=2))
