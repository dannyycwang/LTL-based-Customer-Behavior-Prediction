"""Evaluation gated on all 6 Electronics LR conditions and all 9 neural fits being frozen."""
from common import *
from linear import inputs,bank
from neural import test_predict
from bootstrap_submission_controls import score_function

def gate():
    frozen={}
    for ds in CFG['datasets']:
        for name,path in [('linear',outdir(ds)/'linear_freeze.json')]+[(f'{c}_{s}',outdir(ds)/f'{c}_{s}'/'freeze.json') for c in CFG['neural_conditions'] for s in CFG['neural_seeds']]:
            assert path.exists(),f'Not frozen: {ds}/{name}'
            f=json.loads(path.read_text());assert f['frozen'] and f['config_sha256']==sha(HERE/'config.json')
            model=privdir(ds)/('linear_frozen.pkl' if name=='linear' else f'{name}/best.pt')
            assert sha(model)==f['model_sha256'];frozen[ds+'/'+name]=sha(path)
    frozen['protocol_sha256']=sha(HERE/'PROTOCOL.md')
    frozen['source_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
    save(HERE/'all_models_frozen_before_test.json',frozen)

def evaluate(ds):
    dest=outdir(ds);private=privdir(ds);d=load(private/'prepared.pkl')['test'];bundle=load(private/'linear_frozen.pkl')
    probs={};thresholds={};rows=[];paired=[];start=time.perf_counter()
    for c in CFG['conditions']:
        tick=time.perf_counter();probs['LR_'+c]=bundle['conditions'][c]['model'].predict_proba(inputs(ds,d,bundle,c))[:,1];thresholds['LR_'+c]=bundle['conditions'][c]['threshold']
        timing(ds,'inference',time.perf_counter()-tick,model='LR',condition=c,boundary='loaded R/S/T plus encoding/scaling/logic checking and prediction; timestamp recovery excluded')
    for seed in CFG['neural_seeds']:
        for c in CFG['neural_conditions']:
            tick=time.perf_counter();key=f'NN_{c}_{seed}';probs[key],thresholds[key]=test_predict(ds,c,seed,d,bundle)
            timing(ds,'inference',time.perf_counter()-tick,model='NN',condition=c,seed=seed,boundary='loaded R/S/T plus model load/encoding/logic checking and prediction')
    y=d['label'];groups=pd.factorize(d['user_id'] if ds in ['Cosmetics','Electronics'] else d['session_id'])[0];ng=groups.max()+1
    funcs={};point={};samples={k:[] for k in probs}
    for k,p in probs.items():
        assert np.isfinite(p).all();score=score_function(y,p,thresholds[k]);err=(y-p)**2
        funcs[k]=lambda w,score=score,err=err:np.r_[score(w),np.dot(w,err)/w.sum()]
        point[k]=funcs[k](np.ones(len(y)))
        np.testing.assert_allclose(point[k],[matthews_corrcoef(y,p>=thresholds[k]),average_precision_score(y,p),brier_score_loss(y,p)],atol=1e-12)
    rng=np.random.default_rng(CFG['bootstrap_seed'])
    for _ in range(CFG['bootstrap_replicates']):
        w=np.bincount(rng.integers(ng,size=ng),minlength=ng)[groups]
        for k,f in funcs.items():samples[k].append(f(w))
    samples={k:np.array(v) for k,v in samples.items()}
    for k in probs:
        parts=k.split('_');model,c=parts[:2];seed=int(parts[2]) if len(parts)>2 else 42
        for j,metric in enumerate(['MCC','AP','Brier']):
            lo,hi=np.nanquantile(samples[k][:,j],[.025,.975]);rows.append(dict(dataset=ds,model=model,condition=c,seed=seed,metric=metric,value=point[k][j],lower_95=lo,upper_95=hi,threshold=thresholds[k],n=len(y),positive=int(y.sum())))
    def contrast(a,b,model,contrast,seed):
        delta=point[a]-point[b];values=samples[a]-samples[b]
        for j,metric in enumerate(['MCC','AP','Brier']):
            lo,hi=np.nanquantile(values[:,j],[.025,.975]);paired.append(dict(dataset=ds,model=model,contrast=contrast,seed=seed,metric=metric,delta=delta[j],lower_95=lo,upper_95=hi,unit='user' if ds in ['Cosmetics','Electronics'] else 'session',units=int(ng),replicates=1500,valid_replicates=int(np.isfinite(values[:,j]).sum())))
    for ct in [CFG['primary_contrast']]+CFG['secondary_contrasts']:
        a,b=ct.split('-');contrast('LR_'+a,'LR_'+b,'LR',ct,42)
    for c in CFG['neural_conditions']:
        ks=[f'NN_{c}_{s}' for s in CFG['neural_seeds']];point['MEAN_'+c]=np.mean([point[k] for k in ks],axis=0);samples['MEAN_'+c]=np.mean([samples[k] for k in ks],axis=0)
    for ct in ['M4-M3','M4-M2','M3-M2']:
        a,b=ct.split('-')
        for s in CFG['neural_seeds']:contrast(f'NN_{a}_{s}',f'NN_{b}_{s}','NN',ct,s)
        contrast('MEAN_'+a,'MEAN_'+b,'NN_mean',ct,'mean')
    pd.DataFrame(dict(session_id=d['session_id'],user_id=d['user_id'],label=y,**probs)).to_csv(private/'predictions.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False);pd.DataFrame(paired).to_csv(dest/'paired_differences.csv',index=False)
    save(dest/'evaluation_audit.json',dict(shared_rows_labels=True,shared_resamples_all_seeds=True,independent_units=int(ng),prediction_sha256=sha(private/'predictions.csv.gz'),all_models_frozen_manifest_sha256=sha(HERE/'all_models_frozen_before_test.json'),seconds=time.perf_counter()-start))
    print(ds,'evaluation complete',flush=True)

if __name__=='__main__':
    gate()
    for ds in CFG['datasets']:evaluate(ds)
