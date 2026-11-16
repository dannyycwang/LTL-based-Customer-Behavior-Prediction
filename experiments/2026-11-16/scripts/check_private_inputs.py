"""Check the fingerprinted prepared data needed by the training runners."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config',type=Path)
ap.add_argument('--hashes',action='store_true',help='Read and hash all specified input files.')
args=ap.parse_args()
if args.config:os.environ['LTLF_PATHS']=str(args.config.resolve())
sys.path.insert(0,str(ROOT/'replay'))
from artifact_paths import WORK_DIR,AUDIT_ROOT,COVEO_DATA,COVEO_BASIS,ELECTRONICS_BASIS,COSMETICS_BASIS
checks=[]
def add(p,expected=None):checks.append((p,expected))
def read(rel):return json.loads((ROOT/'provenance/input_manifests'/rel).read_text(encoding='utf-8'))
cov=json.loads((ROOT/'results/runs/protected_first_v1_seed42_short_raw/config.json').read_text())
add(COVEO_DATA,cov['dataset_sha256']);add(COVEO_BASIS,cov['basis_sha256'])
add(AUDIT_ROOT/'reconstructed_split_seed_42.npz')
el=read('electronics_rebuild/seed42_feature_manifest.json')
elroot=WORK_DIR/'electronics_rebuild'
add(ELECTRONICS_BASIS,el['basis_sha256'])
add(elroot/'electronics_ready_noleak/electronics_sessions_ltl_ready.csv',read('electronics_rebuild/manifest.json')['sessions_sha256'])
for split in ['train','val','test']:
    add(elroot/'electronics_ready_noleak/electronics_splits/seed_42'/f'{split}.csv',el['split_hashes']['42_'+split])
for name,h in el['feature_hashes'].items():add(elroot/'seed42_act_rel_features'/name,h)
co=read('cosmetics_pre_purchase_seed42_comparison/feature_manifest.json')
coroot=WORK_DIR/'cosmetics_pre_purchase_seed42'
add(COSMETICS_BASIS,co['basis_sha256'])
for split in ['train','val','test']:add(coroot/f'{split}.csv',co['split_sha256'][split])
for name,entry in co['features'].items():
    filename=entry['path'].replace('\\','/').split('/')[-1]
    add(coroot/'features'/name/filename,entry['sha256'])
add(coroot/'features/counts_scaler.npz')
problems=[]
for p,expected in checks:
    if not p.is_file():problems.append('Missing: '+str(p));continue
    if args.hashes and expected:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:problems.append('Hash differs: '+str(p))
if problems:
    print('\n'.join(problems));raise SystemExit(1)
print(f'{len(checks)} prepared-input paths verified'+(' with available SHA-256 fingerprints.' if args.hashes else '.'))
