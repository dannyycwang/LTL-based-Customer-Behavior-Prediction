"""Exactly nine Electronics fits; inherited encoders/fusion, shared raw timing adapter."""
from common import *
from linear import bank
import torch,random,ast
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from run_coveo import load_originals,set_seed
from run_act_rel_seed42 import original,atomic_checkpoint,rng_state,restore_rng

# Extract the unchanged established encode and fusion classes, avoiding old path globals.
source=HERE.parent/'2026-11-17/neural.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Matched','encode']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'))

class TimedMatched(Matched):
    def __init__(self,ds,sizes,zdim):
        super().__init__(ds,sizes,zdim)
        self.time_projection=nn.Linear(20,64)
    def forward(self,x,z,timing):
        if self.dataset=='Coveo':h=self.raw.encode(x[:,:,0],x[:,:,1],x[:,:,2])
        elif self.dataset=='Electronics':h=self.raw(dict(action=x[:,:,0],product=x[:,:,1],category=x[:,:,2],mask=torch.ones_like(x[:,:,0],dtype=torch.float32)))
        else:h=self.raw(x[:,:,0],torch.zeros_like(x[:,:,0],dtype=torch.bool))
        h=h+self.time_projection(timing)
        return self.head(torch.cat([h,self.summary(z)],dim=1)).squeeze(-1)

@torch.no_grad()
def predict(model,loader,device):
    model.eval();out=[]
    for x,z,t,y in loader:out.append(torch.sigmoid(model(x.to(device),z.to(device),t.to(device))).cpu().numpy())
    return np.concatenate(out)

def features(ds,data,bundle,c):
    z={s:np.column_stack([d['summary'],d['time']]) for s,d in data.items()}
    if c!='M2':
        f=bundle['conditions'][c]
        if f['selected']:
            props,_,scope=bank(ds,c);trees=[freeze(f['rows'][j]['formula_tree']) for j in f['selected']]
            for s,d in data.items():z[s]=np.column_stack([z[s],oldfit.bits(d[scope],props,trees)])
    return z

def train_one(ds,c,seed,data,packed,sizes,vocabs,bundle):
    name=f'{c}_{seed}';dest=outdir(ds)/name;dest.mkdir(exist_ok=True);private=privdir(ds)/name;private.mkdir(exist_ok=True)
    if (dest/'freeze.json').exists():return
    proto=CFG['neural'];z=features(ds,data,bundle,c);scaler=StandardScaler().fit(z['train'])
    z={s:scaler.transform(v).astype(np.float32) for s,v in z.items()}
    ts=StandardScaler().fit(data['train']['time'][:,:20]);tm={s:ts.transform(d['time'][:,:20]).astype(np.float32) for s,d in data.items()}
    set_seed(seed);torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TimedMatched(ds,sizes,z['train'].shape[1]).to(device)
    config=dict(dataset=ds,condition=c,seed=seed,protocol_sha256=sha(HERE/'config.json'),runner_sha256=sha(Path(__file__)),original_neural_sha256=sha(source),parameters=sum(p.numel() for p in model.parameters()),summary_dim=z['train'].shape[1],raw_timing_dim=20,device=str(device),torch=torch.__version__,cuda=torch.version.cuda,**proto)
    save(dest/'config.json',config)
    loaders={s:DataLoader(TensorDataset(torch.from_numpy(packed[s]),torch.from_numpy(z[s]),torch.from_numpy(tm[s]),torch.tensor(d['label'],dtype=torch.float32)),batch_size=proto['batch_size'],shuffle=s=='train') for s,d in data.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=proto['learning_rate'],weight_decay=proto['weight_decay'])
    y=data['train']['label'];lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum())/y.sum()],device=device,dtype=torch.float32))
    start=0;best=-1.;bad=0;history=[];best_state=None;best_epoch=0;ckpath=private/'last.pt'
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['config']==config
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
        start,best,bad,history,best_state,best_epoch=[ck[k] for k in ['epoch','best','bad','history','best_state','best_epoch']]
    for epoch in range(start+1,proto['epochs']+1):
        if bad>=proto['patience']:break
        begin=time.perf_counter();model.train();losses=[]
        for x,zv,t,yv in loaders['train']:
            opt.zero_grad(set_to_none=True);loss=lossfn(model(x.to(device),zv.to(device),t.to(device)),yv.to(device))
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        p=predict(model,loaders['val'],device);score=average_precision_score(data['val']['label'],p)
        if score>best:best=score;bad=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        history.append(dict(epoch=epoch,validation_AP=score,loss=float(np.mean(losses)),seconds=time.perf_counter()-begin))
        atomic_checkpoint(ckpath,dict(config=config,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,best=best,bad=bad,history=history,best_state=best_state,best_epoch=best_epoch))
        save(dest/'epochs.json',history);print(ds,name,history[-1],flush=True)
    model.load_state_dict(best_state);th=threshold(data['val']['label'],predict(model,loaders['val'],device))
    atomic_checkpoint(private/'best.pt',best_state);dump(private/'encoding.pkl',dict(scaler=scaler,time_scaler=ts,vocabs=vocabs,sizes=sizes,zdim=z['train'].shape[1],threshold=th))
    save(dest/'freeze.json',dict(frozen=True,threshold=th,best_epoch=best_epoch,epochs=len(history),parameters=config['parameters'],config_sha256=sha(HERE/'config.json'),model_sha256=sha(private/'best.pt'),encoding_sha256=sha(private/'encoding.pkl'),linear_sha256=sha(privdir(ds)/'linear_frozen.pkl'),test_evaluated=False))
    timing(ds,'neural_training',sum(h['seconds'] for h in history),condition=c,seed=seed,boundary='cached inputs; training plus validation, excludes preprocessing/mining/selection')

def train_all(ds):
    data=load(privdir(ds)/'prepared.pkl');del data['test'];bundle=load(privdir(ds)/'linear_frozen.pkl')
    packed,sizes,vocabs=encode(data,ds)
    for seed in CFG['neural_seeds']:
        for c in CFG['neural_conditions']:train_one(ds,c,seed,data,packed,sizes,vocabs,bundle)

def test_predict(ds,c,seed,d,bundle):
    private=privdir(ds)/f'{c}_{seed}';e=load(private/'encoding.pkl')
    x=d['raw'].reshape(-1,3,10).transpose(0,2,1)
    if ds=='Cosmetics':x=np.array([['|'.join(row) for row in seq] for seq in x],object)[:,:,None]
    packed=np.stack([np.array([[v.get(t,1) for t in row] for row in x[:,:,j]],np.int64) for j,v in enumerate(e['vocabs'])],axis=2)
    z=e['scaler'].transform(features(ds,{'test':d},bundle,c)['test']).astype(np.float32)
    tm=e['time_scaler'].transform(d['time'][:,:20]).astype(np.float32)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=TimedMatched(ds,e['sizes'],e['zdim']).to(device)
    model.load_state_dict(torch.load(private/'best.pt',map_location=device,weights_only=True))
    loader=DataLoader(TensorDataset(torch.from_numpy(packed),torch.from_numpy(z),torch.from_numpy(tm),torch.tensor(d['label'],dtype=torch.float32)),batch_size=CFG['neural']['batch_size'])
    return predict(model,loader,device),e['threshold']

if __name__=='__main__':train_all(sys.argv[1])
