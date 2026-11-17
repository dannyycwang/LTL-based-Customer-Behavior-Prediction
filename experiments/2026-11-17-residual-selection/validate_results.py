"""Final numerical/sample-alignment validation; no model selection."""
from pathlib import Path
import json
import sys
sys.path.insert(0,'C:/Users/ycw/Documents/Codex/2026-11-10/new-chat/work/python_packages')
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,matthews_corrcoef

p=Path(__file__).resolve().parent
cases=[('coveo_v2','coveo_v1',38490,12202),('electronics_v2','electronics_v1',21062,759),
       ('cosmetics_v3','cosmetics_v3',25836,4185)]
checks=[]
for run,cache,nfull,nfixed in cases:
    root=p/'results'/run
    arrays={}
    for split in ['train','val','test']:
        with np.load(p/'results'/cache/f'{split}_features.npz',allow_pickle=True) as z:
            arrays[split]={k:z[k] for k in z.files}
    ids={s:set(t['session_id'].astype(str)) for s,t in arrays.items()}
    assert all(len(ids[s])==len(arrays[s]['session_id']) for s in ids)
    assert not ids['train']&ids['val'] and not ids['train']&ids['test'] and not ids['val']&ids['test']
    for protocol,n in [('full_observation',nfull),('next_10',nfixed)]:
        folder=root/protocol
        metric=pd.read_csv(folder/'metrics.csv')
        dataset=metric.dataset.iloc[0]
        pred=pd.read_csv(folder/f'{dataset.lower()}_{protocol}_predictions.csv.gz',dtype={'session_id':str})
        assert len(pred)==n and pred.session_id.is_unique
        for row in metric.itertuples():
            score=pred[row.condition]
            assert score.between(0,1).all()
            assert abs(average_precision_score(pred.label,score)-row.AUPRC)<1e-12
            assert abs(matthews_corrcoef(pred.label,score>=row.threshold)-row.MCC)<1e-12
        s=json.loads((folder/'selection.json').read_text())
        assert len(s['indices'])==len(set(s['indices']))==10
        t=arrays['train'];keep=np.ones(len(t['length']),bool) if protocol=='full_observation' else t['length']>=10
        f=t['candidate_full' if protocol=='full_observation' else 'candidate_prefix'][keep][:,s['indices']]
        assert np.all(np.ptp(f,axis=0)>0)
        assert len({np.packbits(f[:,i]).tobytes() for i in range(10)})==10
        bootstrap=pd.read_csv(folder/'bootstrap.csv')
        assert (bootstrap.valid_replicates==1500).all()
        checks.append(dict(dataset=dataset,protocol=protocol,n_test=n,conditions=len(metric),status='passed'))
(p/'final_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
