"""One training-only, prefix-aligned TLTLf candidate pool per dataset."""
from pilot import *
import shutil,random,importlib.util

def main(dataset):
    out=HERE/dataset;dest=out/'mining';dest.mkdir(exist_ok=True)
    if (dest/'candidates.json').exists():
        print('Reusing completed single mining run');return
    with (out/'prepared.pkl').open('rb') as f:data=pickle.load(f)
    props=json.loads((out/'preparation.json').read_text())['props'];recipe=CFG['mining'];spec=recipe[dataset]
    rng=np.random.RandomState(CFG['mining_seed']);records=[];traces={0:[],1:[]};counts=[]
    # Fixed equal per-landmark quotas, without replacement; scarce strata are not topped up.
    for t in CFG['landmarks']:
        d=data[t]['train']
        for y in (1,0):
            available=np.flatnonzero(d['label']==y);n=min(len(available),spec['per_class_budget']//3)
            ix=rng.choice(available,n,replace=False)
            for i in ix:
                records.append(dict(session_id=d['session_id'][i],landmark=t,label=y))
                traces[y].append([sorted(s) for s in d['full'][i]])
            counts.append(dict(landmark=t,label=y,available=len(available),sampled=n))
    pd.DataFrame(records).to_csv(dest/'mining_prefix_ids.csv.gz',index=False)
    pd.DataFrame(counts).to_csv(dest/'sampling_counts.csv',index=False)
    obj=dict(vocab=props,traces_pos=traces[1],traces_neg=traces[0],ltlftree='')
    save(dest/'train.json',obj)
    import gc
    del data,traces
    gc.collect()
    # Reuse the exact locally validated fused-gradient learner from the prior follow-up.
    source=WORK/'next_round_20261116_v1/cosmetics_source/TLTLf';vendor=HERE/'vendor';vendor.mkdir(exist_ok=True)
    for name in ('data.py','model.py','interpret.py','train.py'):
        target=vendor/name
        if not target.exists():
            code=(source/name).read_text(encoding='utf-8')
            code=code.replace('from torch.utils.tensorboard import SummaryWriter','')
            code=code.replace('writer = SummaryWriter(log_file)','writer = None')
            code='\n'.join(line for line in code.split('\n') if 'writer.add_scalar(' not in line and 'writer.close()' not in line)
            target.write_text(code,encoding='utf-8')
    sys.path.insert(0,str(vendor))
    import torch
    from train import train,get_train_argument
    from interpret import matrix2ltl
    torch.set_num_threads(1);random.seed(CFG['mining_seed']);np.random.seed(CFG['mining_seed']);torch.manual_seed(CFG['mining_seed'])
    manifest=dict(dataset=dataset,seed=CFG['mining_seed'],recipe=spec,sampling=counts,
        input_sha256=sha(dest/'train.json'),config_sha256=sha(HERE/'config.json'),
        learner_sources={n:sha(vendor/n) for n in ('data.py','model.py','interpret.py','train.py')},
        source='prior Cosmetics fused-gradient TLTLf; same AdamW, regularization and checkpoint rule; CPU',
        scope='entire 1..t prefixes, t=10/20/30, training only; no validation ranking; no old candidate additions',
        status='started')
    save(dest/'manifest.json',manifest);start=time.perf_counter()
    args=get_train_argument(['--train_file',str(dest/'train.json'),'--save_model',str(dest/'model.pt'),
        '--log_file',str(dest/'log'),'--netlen',str(recipe['netlen']),'--epoch',str(spec['epochs']),
        '--lr',str(recipe['lr']),'--batch_size',str(recipe['batch_size'])])
    train(args);train_seconds=time.perf_counter()-start
    trees=matrix2ltl(str(dest/'model.pt'),str(dest/'train.json'),spec['max_candidates'])
    # Keep extraction order. Deduplication by training satisfaction happens separately in B and C.
    rows=[dict(formula_tree=t,formula_str=repr(t),source='candidate_export',candidate=i) for i,t in enumerate(trees)]
    save(dest/'candidates.json',rows)
    manifest.update(status='complete',train_seconds=train_seconds,total_seconds=time.perf_counter()-start,candidates=len(rows),checkpoint_sha256=sha(dest/'model.pt'),candidates_sha256=sha(dest/'candidates.json'))
    save(dest/'manifest.json',manifest);print(manifest,flush=True)

if __name__=='__main__':main(sys.argv[1])
