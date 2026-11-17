"""Real-data GPU execution/shape smoke test, NOT a baseline reproduction or reported result."""
from pathlib import Path
import ast,json,time,math,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from diagnose_coveo import vector,freeze

ROOT=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/ycw/Downloads/product_rawid_vs_ltl_experiment.py')
DATA=Path('C:/Users/ycw/Downloads/browsing_draw_random4_sessions_product_compact.csv')

def main():
    torch.manual_seed(42);np.random.seed(42)
    if not torch.cuda.is_available():raise RuntimeError('GPU not available; do not silently benchmark another device')
    device=torch.device('cuda');torch.set_num_threads(4)
    scope={'nn':nn,'torch':torch}
    mod=ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes=[n for n in mod.body if isinstance(n,ast.ClassDef) and n.name in ('APCTransformer','APCLateFusionTransformer')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
    inds=np.load(ROOT/'reconstructed_split_seed_42.npz')['train'][:4096]
    df=pd.read_csv(DATA,usecols=['session_id_hash','action_sequence','product_sku_hash_sequence','category_hash_sequence','logic_trace_compact','purchase']).iloc[inds].copy()
    arrays=[]
    for col in ['action_sequence','product_sku_hash_sequence','category_hash_sequence']:
        seqs=[json.loads(x)[:10] for x in df[col]]
        vocab={v:i+1 for i,v in enumerate(sorted({v for seq in seqs for v in seq}))}
        a=np.zeros((len(df),10),dtype=np.int64)
        for i,seq in enumerate(seqs):a[i,:len(seq)]=[vocab[v] for v in seq]
        arrays.append(torch.tensor(a,device=device))
    records=json.loads(Path('C:/Users/ycw/Downloads/Z_APC_top20_formulas.json').read_text())[:10]
    trees=[freeze(r['formula_tree']) for r in records]
    keep=['b','d','a','r','fp','rp','sp','pc','p2','p3','sc','cc','rc']
    traces=[[set(s)&set(keep) for s in json.loads(x)] for x in df.logic_trace_compact]
    zp=np.stack([vector(t[:10],trees) for t in traces]);zf=np.stack([vector(t,trees) for t in traces])
    counts=np.array([[sum(p in state for state in tr) for p in keep]+[len(tr)] for tr in traces],dtype=np.float32)
    # This smoke subset contains only reconstructed training rows.
    mean=counts.mean(axis=0);sd=counts.std(axis=0);sd[sd==0]=1
    features={'Short raw':None,'Short+Prefix-LTLf':zp,'Short+Counts':(counts-mean)/sd,'Short+Full-LTLf':zf}
    ys=torch.tensor(df.purchase.to_numpy(),dtype=torch.float32,device=device);rows=[]
    for condition,z in features.items():
        torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
        cls=scope['APCTransformer'] if z is None else scope['APCLateFusionTransformer']
        kwargs=dict(action_vocab_size=int(arrays[0].max())+1,product_vocab_size=int(arrays[1].max())+1,category_vocab_size=int(arrays[2].max())+1,max_len=10)
        if z is not None:kwargs['z_dim']=z.shape[1]
        model=cls(**kwargs).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        zt=torch.tensor(z,dtype=torch.float32,device=device) if z is not None else None
        times=[];losses=[]
        for step in range(10):
            sl=slice(step*256,(step+1)*256);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
            logits=model(*[a[sl] for a in arrays]) if zt is None else model(*[a[sl] for a in arrays],zt[sl])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,ys[sl],pos_weight=torch.tensor(4.,device=device))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if step>=2:times.append(elapsed)
            losses.append(loss.item())
        row=dict(condition=condition,batch_size=256,steps=10,warmup_steps=2,measured_steps=8,feature_dim=0 if z is None else z.shape[1],n_params=sum(p.numel() for p in model.parameters()),median_step_sec=float(np.median(times)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,loss_finite=bool(np.isfinite(losses).all()),estimated_compute_only_25epoch_sec=float(np.median(times)*math.ceil(179616/256)*25))
        rows.append(row);print(row,flush=True)
        del model,opt,zt;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'gpu_smoke_timings.csv',index=False)
    meta=dict(status='Execution smoke only; not baseline reproduction; training subset and vocabulary differ from full protocol; compute estimate excludes data loading, validation, full-raw runs and formula learning.',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,seed=42,train_sample_n=len(df),real_session_ids_sha256=hashlib.sha256('\n'.join(df.session_id_hash).encode()).hexdigest(),source_model_path=str(SOURCE),source_model_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),count_propositions=keep)
    (ROOT/'gpu_smoke_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

if __name__=='__main__':main()
