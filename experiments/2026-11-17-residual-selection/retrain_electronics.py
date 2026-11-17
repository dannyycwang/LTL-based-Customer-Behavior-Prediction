"""Rebuild the missing BP pool using the recorded historical training recipe.

Run from any directory: python retrain_electronics.py
Writes a fresh run directory; never replaces an earlier run.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/ycw/Documents/Codex/2026-11-10/new-chat')
AUDIT = SOURCE / 'outputs/research_audit'
REBUILD = SOURCE / 'work/electronics_rebuild'
NOTEBOOK = SOURCE / 'work/electronics_nb_1_cells.txt'
REPO = Path('C:/Users/ycw/TLTLf')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    out = HERE / 'runs' / ('electronics_retrain_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    vendor = out / 'vendor'
    vendor.mkdir()
    provenance = {}
    for name in ('data.py', 'model.py', 'interpret.py'):
        path = REPO / name
        code = path.read_text(encoding='utf-8')
        if name != 'interpret.py':
            code = code.replace('device = torch.device("cpu")',
                                'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
            # Unused optional logging import; training wrapper does not use it.
            code = code.replace('from torch.utils.tensorboard import SummaryWriter', '')
        (vendor / name).write_text(code, encoding='utf-8')
        provenance[name] = {'original_sha256': sha(path), 'copied_sha256': sha(vendor / name)}
    sys.path.insert(0, str(vendor))
    from data import get_data, collate_fn
    from model import Net
    from interpret import matrix2ltl, LTLf
    from torch.utils.data import DataLoader
    text = NOTEBOOK.read_text(encoding='utf-8')
    cell = text.split('CELL 27 SOURCE\n', 1)[1].split('\nOUTPUT\n', 1)[0]
    tree = ast.parse(cell)
    needed = {'seed_everything', 'parse_list', 'make_trace', 'build_pack_json',
              'train_tltlf_model', 'formula_to_str', 'evaluate_formula_on_json'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in needed]
    assert {n.name for n in nodes} == needed
    recipe = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    (out / 'historical_recipe.py').write_text(recipe, encoding='utf-8')
    scope = dict(np=np, pd=pd, torch=torch, json=json, random=random, time=time,
                 get_data=get_data, collate_fn=collate_fn, Net=Net, LTLf=LTLf,
                 DataLoader=DataLoader, FORMULA_SEED=42, LABEL_COL='purchase',
                 SAMPLE_POS=4000, SAMPLE_NEG=4000, TRACE_TAIL_LEN=10,
                 PRODUCT_MODE='semantic',
                 ACTION_TOKENS={'b','d','a','r','view','cart','purchase','remove'},
                 DWELL_TOKENS={'dl','dm','dh','d_short','d_medium','d_long'},
                 DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    exec(compile(recipe, str(out / 'historical_recipe.py'), 'exec'), scope)
    train = REBUILD / 'electronics_ready_noleak/electronics_splits/seed_42/train.csv'
    manifest = json.loads((AUDIT / 'electronics_rebuild/manifest.json').read_text())
    assert sha(train) == manifest['split_hashes']['42_train']
    frame = pd.read_csv(train)
    train_json = out / 'BP_train.json'
    obj = scope['build_pack_json'](frame, 'BP', train_json)
    reference = REBUILD / 'formula_learning_noleak_4000_last10_seed42/BP/BP_train_seed42_pos4000_neg4000_last10.json'
    assert obj == json.loads(reference.read_text()), 'Rebuilt training traces differ'
    sample = pd.concat([frame[frame.purchase == y].sample(n=4000, random_state=42) for y in (1,0)])
    sample[['session_id_hash','purchase']].to_csv(out / 'formula_pool_ids.csv', index=False)
    assert len(sample) == 8000 and sample.session_id_hash.nunique() == 8000
    config = dict(seed=42, netlen=10, epochs=50, lr=.01, batch_size=100,
                  extraction_limit=250, train_sha256=sha(train), json_sha256=sha(train_json),
                  notebook_sha256=sha(NOTEBOOK), sources=provenance,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  pandas_version=pd.__version__, device=str(scope['DEVICE']),
                  status='training', note='New candidate pool, not recovery of the missing historical checkpoint.')
    (out / 'manifest.json').write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    print('RUN', out, flush=True)
    checkpoint = out / 'BP_model.pt'
    stats = scope['train_tltlf_model'](train_json, checkpoint, netlen=10,
                                      epoch_num=50, lr=.01, batch_size=100)
    start = time.perf_counter()
    trees = matrix2ltl(str(checkpoint), str(train_json), 250)
    rows, signatures = [], set()
    for rank, formula in enumerate(trees, 1):
        ev = scope['evaluate_formula_on_json'](formula, train_json)
        signature = ev.pop('signature')
        unique = signature not in signatures
        signatures.add(signature)
        rows.append(dict(formula_id=f'T{rank:03}', pack='BP', rank_from_tltlf=rank,
                         formula_str=scope['formula_to_str'](formula), formula_tree=repr(formula),
                         semantic_unique=unique, **ev))
    candidates = pd.DataFrame(rows)
    candidates.to_csv(out / 'BP_candidates_top250.csv', index=False)
    top = candidates[candidates.semantic_unique].sort_values(
        ['balanced_accuracy','support_gap','rank_from_tltlf'], ascending=[False,False,True]).head(10)
    top.to_csv(out / 'BP_retrained_standalone_top10.csv', index=False)
    assert len(top) == 10, 'Insufficient distinct candidates'
    config.update(status='complete', training=stats, extraction_seconds=time.perf_counter()-start,
                  candidate_count=len(rows), signature_count=len(signatures),
                  checkpoint_sha256=sha(checkpoint), candidate_sha256=sha(out/'BP_candidates_top250.csv'))
    (out / 'manifest.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

if __name__ == '__main__':
    main()
