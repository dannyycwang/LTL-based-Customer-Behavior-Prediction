"""Matched fixed-origin neural experiments. Resume only complete epochs."""
from common import *
import argparse,random,time,copy
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from run_coveo import load_originals,set_seed
from run_act_rel_seed42 import original,atomic_checkpoint,rng_state,restore_rng

class Matched(nn.Module):
    def __init__(self,dataset,sizes,zdim):
        super().__init__();self.dataset=dataset;self.zdim=zdim
        if dataset=='Coveo':
            _,classes=load_originals()
            self.raw=classes['APCTransformer'](action_vocab_size=sizes[0],product_vocab_size=sizes[1],category_vocab_size=sizes[2],max_len=10)
            self.raw.head=nn.Identity()
        elif dataset=='Electronics':
            mod=original('electronics');self.raw=mod.RawEncoder(n_actions=sizes[0],n_products=sizes[1],n_categories=sizes[2],feature_pack='apc_raw',max_len=10)
        else:
            mod=original('cosmetics');self.raw=mod.RawTransformerEncoder(vocab_size=sizes[0],max_len=10,d_model=64,n_heads=4,n_layers=2,dropout=.15,out_dim=64)
        if zdim:self.summary=nn.Sequential(nn.Linear(zdim,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(64+(64 if zdim else 0),64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x,z):
        if self.dataset=='Coveo':h=self.raw.encode(x[:,:,0],x[:,:,1],x[:,:,2])
        elif self.dataset=='Electronics':h=self.raw(dict(action=x[:,:,0],product=x[:,:,1],category=x[:,:,2],mask=torch.ones_like(x[:,:,0],dtype=torch.float32)))
        else:h=self.raw(x[:,:,0],torch.zeros_like(x[:,:,0],dtype=torch.bool))
        if self.zdim:h=torch.cat([h,self.summary(z)],dim=1)
        return self.head(h).squeeze(-1)

def encode(data,dataset):
    # Train-only vocabulary from the observable prefixes, persisted for replay.
    shaped={s:d['raw'].reshape(-1,3,10).transpose(0,2,1) for s,d in data.items()}
    if dataset=='Cosmetics':shaped={s:np.array([['|'.join(row) for row in seq] for seq in x],object)[:,:,None] for s,x in shaped.items()}
    vocabs=[]
    for j in range(shaped['train'].shape[2]):
        vocabs.append({t:i+2 for i,t in enumerate(sorted(set(shaped['train'][:,:,j].flatten())))})
    packed={s:np.stack([np.array([[v.get(t,1) for t in row] for row in x[:,:,j]],np.int64) for j,v in enumerate(vocabs)],axis=2) for s,x in shaped.items()}
    return packed,[len(v)+2 for v in vocabs],vocabs

@torch.no_grad()
def predict(model,loader,device):
    model.eval();out=[]
    for x,z,y in loader:out.append(torch.sigmoid(model(x.to(device),z.to(device))).cpu().numpy())
    return np.concatenate(out)

def run(dataset,condition,seed,data,packed,sizes,vocabs,smoke=False):
    run=HERE/('smoke' if smoke else 'neural')/f'{dataset}_{condition}_{seed}';run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.json').exists():return
    proto=json.loads((HERE/'protocol.json').read_text())['neural']
    config=dict(dataset=dataset,condition=condition,seed=seed,**proto,smoke=smoke,
                protocol_sha256=sha(HERE/'protocol.json'),prepared_sha256=sha(HERE/'prepared'/f'{dataset}.pkl'),
                runner_sha256=sha(Path(__file__)),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config
    save_json(run/'config.json',config);save_json(run/'vocab.json',vocabs)
    z={s:np.zeros((len(d['label']),0),np.float32) for s,d in data.items()}
    if condition!='raw':
        z={s:d['counts'] if condition=='counts' else np.column_stack([d['counts'],d[condition]]) for s,d in data.items()}
        scaler=StandardScaler().fit(z['train']);z={s:scaler.transform(v).astype(np.float32) for s,v in z.items()}
        np.savez(run/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    set_seed(seed);torch.set_num_threads(4);device=torch.device('cuda')
    model=Matched(dataset,sizes,z['train'].shape[1]).to(device)
    loaders={s:DataLoader(TensorDataset(torch.from_numpy(packed[s]),torch.from_numpy(z[s]),torch.tensor(d['label'],dtype=torch.float32)),batch_size=proto['batch_size'],shuffle=s=='train') for s,d in data.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=proto['learning_rate'],weight_decay=proto['weight_decay'])
    y=data['train']['label'];criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum())/y.sum()],device=device,dtype=torch.float32))
    checkpoint=run/'last_checkpoint.pt';best=-1;bad=0;start=0;best_epoch=0;history=[];best_state=None
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['config']==config
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
        start,best,bad,best_epoch,history,best_state=[ck[k] for k in ['epoch','best','bad','best_epoch','history','best_state']]
    for epoch in range(start+1,(1 if smoke else proto['epochs'])+1):
        if bad>=proto['patience']:break
        begin=time.perf_counter();model.train();losses=[]
        for x,zv,yv in loaders['train']:
            opt.zero_grad(set_to_none=True);loss=criterion(model(x.to(device),zv.to(device)),yv.to(device))
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        p=predict(model,loaders['val'],device);score=average_precision_score(data['val']['label'],p)
        if score>best:best=score;best_epoch=epoch;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        history.append(dict(epoch=epoch,val_AP=score,loss=float(np.mean(losses)),seconds=time.perf_counter()-begin))
        atomic_checkpoint(checkpoint,dict(config=config,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,best=best,bad=bad,best_epoch=best_epoch,history=history,best_state=best_state))
        save_json(run/'epochs.json',history)
        print(dataset,condition,seed,history[-1],flush=True)
    model.load_state_dict(best_state);p=predict(model,loaders['val'],device);yv=data['val']['label']
    threshold=max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda t:matthews_corrcoef(yv,p>=t))
    atomic_checkpoint(run/'best_model.pt',best_state)
    if smoke:save_json(run/'metrics.json',dict(smoke=True,validation_AP=best));return
    begin=time.perf_counter();p=predict(model,loaders['test'],device);infer=time.perf_counter()-begin;y=data['test']['label']
    assert np.isfinite(p).all()
    pred=pd.DataFrame({k:data['test'][k] for k in ['session_id','user_id','label'] if k in data['test']})
    pred['probability']=p;pred['threshold']=threshold;pred['seed']=seed;pred['checkpoint']=str(run/'best_model.pt');pred['config']=str(run/'config.json')
    pred.to_csv(run/'predictions.csv.gz',index=False)
    save_json(run/'metrics.json',dict(dataset=dataset,condition=condition,seed=seed,MCC=matthews_corrcoef(y,p>=threshold),AP=average_precision_score(y,p),threshold=threshold,best_epoch=best_epoch,epochs=len(history),training_validation_seconds=sum(r['seconds'] for r in history),inference_precomputed_seconds=infer,n_test=len(y)))
    print('COMPLETE',dataset,condition,seed,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',nargs='+',default=['Coveo','Electronics','Cosmetics']);ap.add_argument('--smoke',action='store_true');ap.add_argument('--seed',nargs='+',type=int,default=[42,43,44]);args=ap.parse_args()
    for dataset in args.dataset:
        data=load_data(dataset)
        if args.smoke:data={s:{k:v[:512] for k,v in d.items()} for s,d in data.items()}
        packed,sizes,vocabs=encode(data,dataset)
        for seed in args.seed:
            for condition in ['raw','counts','transitions','ltlf']:run(dataset,condition,seed,data,packed,sizes,vocabs,args.smoke)
