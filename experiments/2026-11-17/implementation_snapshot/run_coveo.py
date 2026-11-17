"""Coveo Act+Rel reproducible downstream entry point.

Historical compatibility mode keeps reconstructed old splits and last-window input.
Protected mode pins the verified formula-learning/selection pool to training and
uses first-window input in every short condition. No source files are changed.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,math,random,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]/'work'
if (WORK/'python_packages').exists():sys.path.insert(0,str(WORK/'python_packages'))
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import default_collate
from diagnose_coveo import vector,freeze,sha,validate

HOME=Path('C:/Users/ycw')
MODEL_SOURCE=HOME/'Downloads/product_rawid_vs_ltl_experiment.py'
BASE_SOURCE=HOME/'Downloads/ltl_integrated_pytorch_behavior_dwell_automaton_experiment.py'
DATA=HOME/'Downloads/browsing_draw_random4_sessions_product_compact.csv'
BASIS=HOME/'Downloads/Z_APC_top20_formulas.json'
PROPS=['b','d','a','r','fp','rp','sp','pc','p2','p3','sc','cc','rc']
CONDITIONS=['full_raw','short_raw','prefix_ltlf','counts','full_ltlf']

def trim_full_batch(samples):
    """Remove right-padding columns shared by all sessions in the batch."""
    actions, products, categories, labels = default_collate(samples)
    occupied = torch.where(actions.ne(0).any(dim=0))[0]
    width = int(occupied[-1]) + 1 if len(occupied) else 1
    return actions[:, :width], products[:, :width], categories[:, :width], labels

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True

def load_originals():
    spec=importlib.util.spec_from_file_location('original_coveo_evaluator',BASE_SOURCE)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    scope={'nn':nn,'torch':torch}
    mod=ast.parse(MODEL_SOURCE.read_text(encoding='utf-8'))
    nodes=[n for n in mod.body if isinstance(n,ast.ClassDef) and n.name in ('APCTransformer','APCLateFusionTransformer')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(MODEL_SOURCE),'exec'),scope)
    return base,scope

def protected_split(df,seed,old):
    protected=np.zeros(len(df),dtype=bool)
    for y in [0,1]:
        sampled=df[(df['split']=='train')&(df.purchase==y)].sample(n=10000,random_state=42)
        protected[sampled.index]=True
    rng=np.random.RandomState(seed);result={k:[] for k in ['train','val','test']}
    # Match the old split's total and per-class sample sizes, without overlap.
    for y in [0,1]:
        pool=np.flatnonzero((df.purchase.to_numpy()==y)&~protected);rng.shuffle(pool)
        nval=int((df.purchase.to_numpy()[old['val']]==y).sum());ntest=int((df.purchase.to_numpy()[old['test']]==y).sum())
        result['val'].extend(pool[:nval]);result['test'].extend(pool[nval:nval+ntest])
        result['train'].extend(pool[nval+ntest:]);result['train'].extend(np.flatnonzero((df.purchase.to_numpy()==y)&protected))
    for k in result:
        result[k]=np.asarray(result[k],dtype=np.int64);rng.shuffle(result[k])
    assert not protected[result['val']].any() and not protected[result['test']].any()
    assert len(set(np.concatenate(list(result.values()))))==len(df)
    return result,protected

@torch.no_grad()
def predict(model,loader,device):
    model.eval();ys=[];ps=[]
    for batch in loader:
        xs=[x.to(device,non_blocking=True) for x in batch[:-1]]
        ps.append(torch.sigmoid(model(*xs)).cpu().numpy());ys.append(batch[-1].numpy())
    return np.concatenate(ys),np.concatenate(ps)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['coveo'],default='coveo');p.add_argument('--seed',type=int,choices=[42,43,44,45],required=True)
    p.add_argument('--init-seed',type=int,help='Override downstream training RNG while preserving the split and formula basis.')
    p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--protocol',choices=['historical','protected_first_v1'],required=True)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--patience',type=int,default=999);p.add_argument('--max-minutes',type=float,default=30)
    p.add_argument('--out',type=Path,default=ROOT/'runs');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--benchmark-batches',type=int,default=0)
    args=p.parse_args()
    init_seed = args.seed if args.init_seed is None else args.init_seed
    suffix = '' if args.init_seed is None else f'_init{init_seed}'
    run=args.out/f'{args.protocol}_seed{args.seed}_{args.condition}{suffix}';run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.json').exists():print('Completed; retaining existing result:',run);return
    if args.protocol=='historical' and args.condition!='short_raw':raise ValueError('Historical compatibility check is defined for short_raw only')
    cols=['session_id_hash','split','purchase','action_sequence','product_sku_hash_sequence','category_hash_sequence','logic_trace_compact']
    prep_t=time.perf_counter();df=pd.read_csv(DATA,usecols=cols)
    old=dict(np.load(ROOT/f'reconstructed_split_seed_{args.seed}.npz'))
    if args.protocol=='historical':splits=old;protected=np.zeros(len(df),bool)
    else:splits,protected=protected_split(df,args.seed,old)
    for k,idx in splits.items():
        pd.DataFrame({'session_id':df.session_id_hash.iloc[idx],'label':df.purchase.iloc[idx]}).to_csv(run/f'{k}_ids.csv',index=False)
    if args.protocol!='historical':df.loc[protected,['session_id_hash','purchase']].to_csv(run/'protected_formula_pool.csv',index=False)
    records=json.loads(BASIS.read_text())[:10];trees=[freeze(r['formula_tree']) for r in records]
    if args.condition in ['prefix_ltlf','counts','full_ltlf']:validate(trees)
    seqs=[[json.loads(x) for x in df[c]] for c in ['action_sequence','product_sku_hash_sequence','category_hash_sequence']]
    lengths=np.array([len(s) for s in seqs[0]])
    if not all(len(a)==len(b)==len(c) for a,b,c in zip(*seqs)):raise ValueError('Raw event fields are misaligned')
    raw_len=int(lengths.max()) if args.condition=='full_raw' else 10
    mode='last' if args.protocol=='historical' else 'first'
    vocabs=[{'b':1,'d':2,'a':3,'r':4}]
    for seq in seqs[1:]:
        tokens=sorted({str(t) for i in splits['train'] for t in seq[i]})
        v={'<PAD>':0,'<UNK>':1}
        for t in tokens:
            if t not in v:v[t]=len(v)
        vocabs.append(v)
    raw=[]
    for seq,v in zip(seqs,vocabs):
        a=np.zeros((len(df),raw_len),dtype=np.int64)
        for i,s in enumerate(seq):
            sub=s[-raw_len:] if mode=='last' else s[:raw_len]
            a[i,:len(sub)]=[v.get(str(t),1) for t in sub]
        raw.append(a)
    feature_t=time.perf_counter();z=None
    if args.condition in ['prefix_ltlf','counts','full_ltlf']:
        traces=[[set(s)&set(PROPS) for s in json.loads(x)] for x in df.logic_trace_compact]
        if args.condition=='counts':
            z=np.array([[sum(p in s for s in tr) for p in PROPS]+[len(tr)] for tr in traces],dtype=np.float32)
            mean=z[splits['train']].mean(axis=0);std=z[splits['train']].std(axis=0);std[std==0]=1
            np.savez(run/'counts_scaler_train_only.npz',mean=mean,std=std,propositions=np.array(PROPS+['trace_length']))
            z=(z-mean)/std
        else:
            z=np.stack([vector(t[:10] if args.condition=='prefix_ltlf' else t,trees) for t in traces]).astype(np.float32)
        np.save(run/'features.npy',z)
    feature_sec=time.perf_counter()-feature_t
    fingerprint=dict(dataset_sha256=sha(DATA),basis_sha256=sha(BASIS),model_source_sha256=sha(MODEL_SOURCE),evaluator_source_sha256=sha(BASE_SOURCE),runner_sha256=sha(Path(__file__)),splits={k:sha(run/f'{k}_ids.csv') for k in splits},condition=args.condition,protocol=args.protocol,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,patience=args.patience)
    if args.init_seed is not None:
        fingerprint['init_seed'] = init_seed
    config=dict(**fingerprint,raw_window=mode,raw_len=raw_len,feature_dim=0 if z is None else z.shape[1],protected_formula_pool_n=int(protected.sum()),preparation_sec=time.perf_counter()-prep_t,feature_construction_all_splits_sec=feature_sec,formula_learning_sec=0.,formula_cost_status='fixed pre-existing basis; historical formula-learning cost excluded and not measured',historical_compatibility_limit='Old script reset seed AFTER model construction; standalone initialization cannot reproduce RNG state inherited from preceding models.',user_overlap='unverified; user IDs absent',threshold_protocol='original validation F1 grid after checkpoint selection; no test-label threshold tuning')
    if args.condition=='full_raw':
        config['batch_padding']='right-pad to longest session in batch; keep all observed events and original position IDs'
        config['truncated_sessions']=0
    config_path=run/'config.json'
    if config_path.exists():
        prev=json.loads(config_path.read_text())
        if any(prev.get(k)!=v for k,v in fingerprint.items()):raise ValueError('Run configuration differs from existing run; use a new output directory')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    (run/'basis.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (run/'vocabs.json').write_text(json.dumps(vocabs),encoding='utf-8')
    if args.prepare_only:print('Prepared:',run,flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    base,classes=load_originals();device=torch.device('cuda');torch.set_num_threads(4);set_seed(init_seed)
    kwargs=dict(action_vocab_size=5,product_vocab_size=len(vocabs[1]),category_vocab_size=len(vocabs[2]),max_len=raw_len)
    if z is None:model=classes['APCTransformer'](**kwargs)
    else:model=classes['APCLateFusionTransformer'](**kwargs,z_dim=z.shape[1])
    model=model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loaders={}
    for k,idx in splits.items():
        tensors=[torch.from_numpy(a[idx]) for a in raw]
        if z is not None:tensors.append(torch.from_numpy(z[idx]))
        tensors.append(torch.tensor(df.purchase.to_numpy()[idx],dtype=torch.float32))
        loaders[k]=DataLoader(TensorDataset(*tensors),batch_size=args.batch_size,shuffle=k=='train',num_workers=0,pin_memory=True,
                              collate_fn=trim_full_batch if args.condition=='full_raw' else None)
    npos=int(df.purchase.iloc[splits['train']].sum());weight=(len(splits['train'])-npos)/npos
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device=device))
    if args.benchmark_batches:
        def train_step(batch):
            opt.zero_grad(set_to_none=True)
            inputs = [field.to(device, non_blocking=True) for field in batch]
            loss = criterion(model(*inputs[:-1]), inputs[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            if not torch.isfinite(loss).item():
                raise ValueError('Nonfinite benchmark loss')

        train_step(default_collate([loaders['train'].dataset[i] for i in range(args.batch_size)]))
        torch.cuda.synchronize()
        times = []
        for i, batch in enumerate(loaders['train']):
            begin = time.perf_counter()
            train_step(batch)
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.perf_counter() - begin)
            if len(times) == args.benchmark_batches:
                break
        begin = time.perf_counter()
        predict(model, loaders['val'], device)
        torch.cuda.synchronize()
        val_sec = time.perf_counter() - begin
        print(json.dumps(dict(max_length=raw_len, timed_batches=len(times),
                              train_batches=len(loaders['train']), mean_train_batch_sec=float(np.mean(times)),
                              validation_sec=val_sec, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)),
              flush=True)
        return
    start=1;best_score=-1.;best_epoch=-1;best_state=None;bad=0;elapsed_prior=0.;checkpoint=run/'last_checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best_score=ck['best_score'];best_epoch=ck['best_epoch'];best_state=ck['best_state'];bad=ck['bad'];elapsed_prior=ck['elapsed_sec']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    set_limit=args.max_minutes*60;torch.cuda.synchronize();begin=time.perf_counter();epoch=start-1
    for epoch in range(start,args.epochs+1):
        if time.perf_counter()-begin>=set_limit:
            print('Time budget reached; checkpoint retained:',run,flush=True);return
        model.train();losses=[]
        for batch in loaders['train']:
            if time.perf_counter()-begin>=set_limit:
                print('Time budget reached mid-epoch; resume from last completed epoch:',run,flush=True);return
            opt.zero_grad(set_to_none=True);xs=[x.to(device,non_blocking=True) for x in batch[:-1]];target=batch[-1].to(device,non_blocking=True)
            loss=criterion(model(*xs),target);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
        vy,vp=predict(model,loaders['val'],device);score=base.compute_metrics(vy,vp,threshold=.5)['f1']
        if score>best_score:best_score=float(score);best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        torch.cuda.synchronize();elapsed=elapsed_prior+time.perf_counter()-begin
        pending_checkpoint=checkpoint.with_suffix('.tmp')
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),best_score=best_score,best_epoch=best_epoch,best_state=best_state,bad=bad,elapsed_sec=elapsed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),pending_checkpoint)
        pending_checkpoint.replace(checkpoint)
        with (run/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,val_f1_at_05=float(score),mean_train_loss=float(np.mean(losses)),elapsed_sec=elapsed))+'\n')
        print(f'epoch={epoch} val_f1@0.5={score:.6f} elapsed={elapsed:.1f}s',flush=True)
        if bad>=args.patience:break
    model.load_state_dict(best_state);vy,vp=predict(model,loaders['val'],device);threshold,_=base.choose_threshold(vy,vp,metric='f1')
    torch.cuda.synchronize();infer_start=time.perf_counter();ty,tp=predict(model,loaders['test'],device);torch.cuda.synchronize();infer_sec=time.perf_counter()-infer_start
    pred=pd.DataFrame(dict(session_id=df.session_id_hash.iloc[splits['test']].to_numpy(),label=ty.astype(int),probability=tp,length=lengths[splits['test']],threshold=threshold,seed=args.seed,condition=args.condition,protocol=args.protocol))
    pred.to_csv(run/'predictions.csv',index=False);rows=[]
    for label,mask in [('all',np.ones(len(pred),bool)),('L<=10',pred.length.to_numpy()<=10),('L>10',pred.length.to_numpy()>10)]:
        yy=ty[mask];pp=tp[mask];m=base.compute_metrics(yy,pp,threshold=threshold) if len(yy) else {}
        if len(np.unique(yy))<2:
            for name in ['mcc','auprc','auc']:m[name]=None
        rows.append(dict(stratum=label,n=len(yy),positive_rate=float(yy.mean()) if len(yy) else None,**m))
    pd.DataFrame(rows).to_csv(run/'stratified_metrics.csv',index=False)
    metrics=dict(**base.compute_metrics(ty,tp,threshold=threshold),dataset='coveo',seed=args.seed,condition=args.condition,protocol=args.protocol,best_epoch=best_epoch,actual_epochs_ran=epoch,n_params=sum(p.numel() for p in model.parameters()),feature_dim=0 if z is None else z.shape[1],train_including_validation_sec=elapsed_prior+time.perf_counter()-begin-infer_sec,inference_precomputed_features_sec=infer_sec,feature_construction_all_splits_sec=feature_sec,gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda)
    if args.init_seed is not None:
        metrics['init_seed'] = init_seed
    torch.save(best_state,run/'best_model.pt')
    (run/'metrics.tmp').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (run/'metrics.tmp').replace(run/'metrics.json')
    print('COMPLETE',json.dumps(metrics),flush=True)

if __name__=='__main__':main()
