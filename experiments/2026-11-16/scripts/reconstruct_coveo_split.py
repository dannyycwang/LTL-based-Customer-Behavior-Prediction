"""Rebuild notebook cell-15 indices from a hash-matched Coveo CSV, without relearning formulas."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'replay'))
from artifact_paths import AUDIT_ROOT,COVEO_DATA
expected=json.loads((ROOT/'results/runs/protected_first_v1_seed42_short_raw/config.json').read_text())['dataset_sha256']
h=hashlib.sha256()
with COVEO_DATA.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
if h.hexdigest()!=expected:raise ValueError('Coveo CSV does not match the audited dataset.')
y=pd.read_csv(COVEO_DATA,usecols=['purchase']).purchase.to_numpy()
tr,temp=train_test_split(np.arange(len(y)),test_size=.3,random_state=42,stratify=y)
va,te=train_test_split(temp,test_size=.5,random_state=43,stratify=y[temp])
AUDIT_ROOT.mkdir(parents=True,exist_ok=True)
path=AUDIT_ROOT/'reconstructed_split_seed_42.npz'
if path.exists():
    with np.load(path) as prior:
        if not all(np.array_equal(prior[k],v) for k,v in {'train':tr,'val':va,'test':te}.items()):
            raise ValueError('Existing split differs; retained without modification.')
else:np.savez_compressed(path,train=tr,val=va,test=te)
print('Reconstructed historical index sizes:',len(tr),len(va),len(te))
print('run_coveo.py --protocol protected_first_v1 subsequently protects the formula pool.')
