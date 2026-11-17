"""Replay saved checkpoints, thresholds and cohort alignment without tuning."""
from common import *
from neural import Matched,encode,predict
from sklearn.metrics import matthews_corrcoef,average_precision_score
from torch.utils.data import DataLoader,TensorDataset
import torch

def main():
    torch.set_num_threads(4);rows=[];old=pd.read_csv(AUDIT/'submission_iterations_seed42/cohort_counts.csv')
    for ds in ['Coveo','Electronics','Cosmetics']:
        data=load_data(ds);packed,sizes,vocabs=encode(data,ds)
        for split,d in data.items():
            r=old[(old.dataset==ds)&(old.protocol=='next_10')&(old.split==split)].iloc[0]
            assert len(d['label'])==r.n and d['label'].sum()==r.positives
            assert d['raw'].shape==(r.n,30);assert np.isfinite(d['counts']).all()
            assert len(set(d['session_id']))==len(d['label'])
        assert not set(data['train']['session_id'])&set(data['test']['session_id'])
        for seed in [42,43,44]:
            for c in ['raw','counts','transitions','ltlf']:
                run=HERE/'neural'/f'{ds}_{c}_{seed}';metrics=json.loads((run/'metrics.json').read_text());config=json.loads((run/'config.json').read_text());assert not config['smoke']
                z={s:np.zeros((len(d['label']),0),np.float32) for s,d in data.items()}
                if c!='raw':
                    scaler=np.load(run/'scaler.npz');z={s:d['counts'] if c=='counts' else np.column_stack([d['counts'],d[c]]) for s,d in data.items()};z={s:((v-scaler['mean'])/scaler['scale']).astype(np.float32) for s,v in z.items()}
                model=Matched(ds,sizes,z['train'].shape[1]).cuda();model.load_state_dict(torch.load(run/'best_model.pt',weights_only=True))
                ps={}
                for s in ['val','test']:
                    loader=DataLoader(TensorDataset(torch.from_numpy(packed[s]),torch.from_numpy(z[s]),torch.tensor(data[s]['label'],dtype=torch.float32)),batch_size=256,shuffle=False)
                    ps[s]=predict(model,loader,torch.device('cuda'))
                selected=max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda t:matthews_corrcoef(data['val']['label'],ps['val']>=t));assert selected==metrics['threshold']
                saved=pd.read_csv(run/'predictions.csv.gz',dtype={'session_id':str,'user_id':str});assert np.array_equal(saved.session_id,data['test']['session_id'])
                err=float(np.abs(saved.probability.to_numpy()-ps['test']).max());assert err<1e-6
                assert abs(matthews_corrcoef(saved.label,saved.probability>=selected)-metrics['MCC'])<1e-12
                assert abs(average_precision_score(saved.label,saved.probability)-metrics['AP'])<1e-10
                val=pd.DataFrame({k:data['val'][k] for k in ['session_id','user_id','label'] if k in data['val']});val['probability']=ps['val'];val['threshold']=selected;val['seed']=seed;val.to_csv(run/'validation_predictions.csv.gz',index=False)
                rows.append(dict(dataset=ds,condition=c,seed=seed,test_max_abs_probability_difference=err,threshold_replayed=True,cohort_counts_match=True))
                del model
        print('verified',ds,flush=True)
    pd.DataFrame(rows).to_csv(HERE/'audit/run_verification.csv',index=False);assert len(rows)==36
if __name__=='__main__':main()
