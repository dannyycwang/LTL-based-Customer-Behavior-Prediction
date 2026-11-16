"""Install aggregate audit receipts in an ignored local workspace; copy no sessions."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path)
args=parser.parse_args()
if args.config:os.environ['LTLF_PATHS']=str(args.config.resolve())
sys.path.insert(0,str(ROOT/'replay'))
from artifact_paths import AUDIT_ROOT,WORK_DIR,resolve_saved_path

AUDIT_ROOT.mkdir(parents=True,exist_ok=True)
WORK_DIR.mkdir(parents=True,exist_ok=True)
source=ROOT/'provenance/input_manifests'
path_keys={'path','feature_dir','ready_dir'}
def relocate(value,key=None):
    if isinstance(value,dict):return {k:relocate(v,k) for k,v in value.items()}
    if isinstance(value,list):return [relocate(v,key) for v in value]
    if isinstance(value,str) and key in path_keys:return str(resolve_saved_path(value))
    return value
created=0
for p in source.rglob('*.json'):
    target=AUDIT_ROOT/p.relative_to(source)
    if target.exists():continue
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(relocate(json.loads(p.read_text(encoding='utf-8'))),indent=2),encoding='utf-8')
    created+=1
print(f'Created {created} aggregate audit receipts. Existing local receipts were retained.')
print('Supply the corresponding private dataset/split/feature files before running training.')
print(f'Audit outputs: {AUDIT_ROOT}\nPrivate inputs: {WORK_DIR}')
