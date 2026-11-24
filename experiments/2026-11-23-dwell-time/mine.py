from common import *
import random

def main(ds):
    dest=outdir(ds);private=privdir(ds)
    if (dest/'dwell_candidates.json').exists():return
    if (dest/'mining_started.json').exists():
        raise RuntimeError('A mining attempt already exists. Inspect its process/logs; do not silently start a second pool.')
    prep=json.loads((dest/'preparation.json').read_text());manifest=prep['historical_mining_manifest']
    for n,h in manifest['learner_sources'].items():assert sha(PREV/'vendor'/n)==h
    sys.path.insert(0,str(PREV/'vendor'))
    import torch
    from train import train,get_train_argument
    from interpret import matrix2ltl
    torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    spec=manifest['recipe'];recipe=old.CFG['mining'];started=time.perf_counter()
    save(dest/'mining_started.json',dict(seed=42,recipe=spec,input_sha256=sha(private/'mining_train.json'),learner_sources=manifest['learner_sources'],matching_records=True,config_sha256=sha(HERE/'config.json')))
    args=get_train_argument(['--train_file',str(private/'mining_train.json'),'--save_model',str(private/'mining_model.pt'),'--log_file',str(private/'mining_log'),'--netlen',str(recipe['netlen']),'--epoch',str(spec['epochs']),'--lr',str(recipe['lr']),'--batch_size',str(recipe['batch_size'])])
    train(args)
    trees=matrix2ltl(str(private/'mining_model.pt'),str(private/'mining_train.json'),spec['max_candidates'])
    rows=[dict(formula_tree=t,formula_str=repr(t),source='dwell_learned',candidate=i) for i,t in enumerate(trees)]
    save(dest/'dwell_candidates.json',rows)
    save(dest/'mining_complete.json',dict(seconds=time.perf_counter()-started,candidates=len(rows),checkpoint_sha256=sha(private/'mining_model.pt'),candidates_sha256=sha(dest/'dwell_candidates.json')))
    timing(ds,'mining',time.perf_counter()-started)
if __name__=='__main__':main(sys.argv[1])
