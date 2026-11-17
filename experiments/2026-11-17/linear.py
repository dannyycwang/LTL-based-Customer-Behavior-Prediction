from common import *
import time,joblib
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,matthews_corrcoef
from scipy.sparse import hstack,csr_matrix

def select(bits,y):
    pos=bits[y==1].mean(0);neg=bits[y==0].mean(0);support=bits.sum(0)
    order=sorted(np.flatnonzero((support>=20)&(support<=len(y)-20)),key=lambda j:(-abs(pos[j]-neg[j]),int(j)))
    seen=set();picked=[]
    for j in order:
        key=np.packbits(bits[:,j]).tobytes()
        if key in seen:continue
        seen.add(key);picked.append(int(j))
        if len(picked)==10:break
    return picked

def fit(dataset,data,zs,out,tag):
    encoder=OneHotEncoder(handle_unknown='ignore');raw={'train':encoder.fit_transform(data['train']['raw'])}
    for s in ['val','test']:raw[s]=encoder.transform(data[s]['raw'])
    for name in zs['train']:
        run=out/name;run.mkdir(parents=True,exist_ok=True)
        if (run/'metrics.json').exists():continue
        scaler=StandardScaler().fit(zs['train'][name]);x={s:hstack([raw[s],csr_matrix(scaler.transform(zs[s][name]))],format='csr') for s in data}
        model=LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=42)
        start=time.perf_counter();model.fit(x['train'],data['train']['label']);secs=time.perf_counter()-start
        assert model.n_iter_[0]<2000
        pv=model.predict_proba(x['val'])[:,1];yv=data['val']['label']
        threshold=max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda t:matthews_corrcoef(yv,pv>=t))
        p=model.predict_proba(x['test'])[:,1];y=data['test']['label']
        pred=pd.DataFrame({k:data['test'][k] for k in ['session_id','user_id','label'] if k in data['test']});pred['probability']=p;pred['threshold']=threshold;pred['seed']=42
        pred.to_csv(run/'predictions.csv.gz',index=False)
        joblib.dump(dict(model=model,scaler=scaler,encoder=encoder,threshold=threshold),run/'model.joblib')
        save_json(run/'config.json',dict(dataset=dataset,condition=name,protocol=tag,protocol_sha256=sha(HERE/'protocol.json'),runner_sha256=sha(Path(__file__)),C=1,solver='liblinear',max_iter=2000,selection='validation MCC grid'))
        save_json(run/'metrics.json',dict(dataset=dataset,condition=name,protocol=tag,MCC=matthews_corrcoef(y,p>=threshold),AP=average_precision_score(y,p),threshold=threshold,training_seconds=secs,n_test=len(y)))
        print('linear complete',dataset,name,flush=True)

def main(dataset):
    data=load_data(dataset);out=HERE/'linear'/dataset;out.mkdir(parents=True,exist_ok=True)
    selections={key:select(data['train'][key],data['train']['eventual_label']) for key in ['declare_all','transition_all','logic_all']}
    save_json(out/'selection.json',dict(selections=selections,target='eventual_purchase',ranking_ids=str(HERE/'prepared'/f'{dataset}_train_ids.csv'),duplicate_policy='identical train bit columns removed; this is not logical equivalence',unavailable_comparisons=[k for k,v in selections.items() if len(v)<10]))
    assert len(selections['declare_all'])==len(selections['transition_all'])==10
    zs={}
    for s,d in data.items():
        zs[s]={'counts':d['counts'],'ltlf_frozen':np.column_stack([d['counts'],d['ltlf']]),
               'transitions':np.column_stack([d['counts'],d['transition_all'][:,selections['transition_all']]]),
               'declare':np.column_stack([d['counts'],d['declare_all'][:,selections['declare_all']]])}
        if len(selections['logic_all'])==10:zs[s]['ltlf_matched_rank']=np.column_stack([d['counts'],d['logic_all'][:,selections['logic_all']]])
    fit(dataset,data,zs,out,'event10_H10_template_control')
if __name__=='__main__':
    for d in sys.argv[1:] or ['Coveo','Electronics','Cosmetics']:main(d)
