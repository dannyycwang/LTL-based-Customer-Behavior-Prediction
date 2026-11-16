"""Reconstruct notebook split/sample choices and verify against stored training JSON."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT = AUDIT_ROOT
WORK=WORK_DIR
sys.path.insert(0,str(WORK/'python_packages'))
from sklearn.model_selection import train_test_split
import sklearn

def main():
    p=COVEO_DATA
    d=pd.read_csv(p,usecols=['session_id_hash','purchase','split','logic_trace_compact'])
    old=json.loads((WORK/'coveo_formula_training.json').read_text(encoding='utf-8'))
    keep=set(old['vocab']);sample_ids=[];checks=[]
    for label,key in [(1,'traces_pos'),(0,'traces_neg')]:
        sample=d[(d['split']=='train')&(d.purchase==label)].sample(n=10000,random_state=42)
        traces=[[[a for a in s if a in keep] for s in json.loads(x)[-30:]] for x in sample.logic_trace_compact]
        matches=[a==b for a,b in zip(traces,old[key])]
        checks.append(dict(label=label,n=len(traces),positionwise_exact_trace_matches=sum(matches)))
        if len(traces)!=len(old[key]) or not all(matches):
            raise ValueError(f'Formula sample provenance not reproduced: {checks[-1]}')
        sample_ids.extend(sample.session_id_hash.tolist())
    formula_ids=set(sample_ids);rows=[]
    for seed in [42,43,44,45]:
        tr,temp=train_test_split(np.arange(len(d)),test_size=.3,random_state=seed,stratify=d.purchase.to_numpy())
        va,te=train_test_split(temp,test_size=.5,random_state=seed+1,stratify=d.purchase.to_numpy()[temp])
        for split,idx in [('train',tr),('val',va),('test',te)]:
            ids=d.session_id_hash.iloc[idx].astype(str);overlap=ids.isin(formula_ids)
            rows.append(dict(seed=seed,downstream_split=split,n=len(idx),formula_training_overlap_n=int(overlap.sum()),fraction_of_downstream_split=float(overlap.mean()),fraction_of_formula_training_sample=float(overlap.sum()/len(formula_ids)),split_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
        np.savez_compressed(ROOT/f'reconstructed_split_seed_{seed}.npz',train=tr,val=va,test=te)
    pd.DataFrame(rows).to_csv(ROOT/'coveo_formula_training_overlap.csv',index=False)
    (ROOT/'coveo_formula_sample_provenance.json').write_text(json.dumps(dict(status='Formula training sample exact trace order verified against stored Drive JSON; downstream indices reconstructed from notebook cell 15, not original saved ID lists.',sklearn_version=sklearn.__version__,formula_training_trace_checks=checks,n_formula_training_ids=len(formula_ids),formula_input_archive='private formula-training archive'),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
