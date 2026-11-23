"""Sparse raw+summary conditional selection, then frozen paired evaluation."""
from pilot import *
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,average_precision_score,matthews_corrcoef,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor
from bootstrap_submission_controls import score_function
import warnings
warnings.simplefilter('error',ConvergenceWarning)

def bits(traces,props,trees):
    """Batch finite-trace DP with shared subtrees. Strong next and strong until."""
    x=np.array([[[p in s for p in props] for s in tr] for tr in traces],bool)
    n,L,_=x.shape;cache={p:x[:,:,i] for i,p in enumerate(props)}
    cache['true']=np.ones((n,L),bool);cache['false']=np.zeros((n,L),bool)
    def run(t):
        t=freeze(t)
        if t in cache:return cache[t]
        op=t[0];a=run(t[1]);v=np.zeros((n,L),bool)
        if op in ('!','not'):v=~a
        elif op=='X':v[:,:-1]=a[:,1:]
        elif op=='F':v=np.logical_or.accumulate(a[:,::-1],axis=1)[:,::-1]
        elif op=='G':v=np.logical_and.accumulate(a[:,::-1],axis=1)[:,::-1]
        else:
            b=run(t[2])
            if op in ('&','&&','and'):v=a&b
            elif op in ('|','||','or'):v=a|b
            elif op=='->':v=~a|b
            elif op=='U':
                v[:,-1]=b[:,-1]
                for i in range(L-2,-1,-1):v[:,i]=b[:,i]|(a[:,i]&v[:,i+1])
            else:raise ValueError(op)
        cache[t]=v;return v
    result=np.stack([run(t)[:,0] for t in trees],axis=1).astype(np.uint8)
    for i in range(min(12,n)):
        assert np.array_equal(result[i],vector(traces[i],trees)),'Independent checker mismatch'
    return result

def model(x,y,initial=None):
    from native_logistic import NativeLogistic
    return NativeLogistic(C=CFG['classifier']['C'],tol=CFG['native_tolerance']).fit(x,y,initial=initial)

def augment(x,z,selected):return hstack([x,csr_matrix(z[:,selected])],format='csr') if selected else x

def select(bt,bv,ft,fv,yt,yv,rows,out,baseline_signatures,baseline_model=None):
    """Minimal adaptation of 2026-11-17-residual-selection/run_residual.py::select.
    Preserve greedy validation logloss/refit/tie rule. Condition on full A, use sparse
    raw columns, remove constants/duplicates, stop before accepting nonpositive gain.
    """
    out.mkdir(exist_ok=True);start=time.perf_counter();excluded=[];eligible=[];seen=set(baseline_signatures)
    fs=StandardScaler().fit(ft);zt,zv=fs.transform(ft),fs.transform(fv)
    for j in range(ft.shape[1]):
        signature=np.flatnonzero(ft[:,j]).astype(np.int64).tobytes()
        reason='training_constant' if np.ptp(ft[:,j])==0 else ('exact_training_duplicate_candidate_or_A' if signature in seen else None)
        if reason:excluded.append(dict(candidate=j,reason=reason,formula=rows[j]['formula_str']))
        else:seen.add(signature);eligible.append(j)
    pd.DataFrame(excluded).to_csv(out/'excluded.csv',index=False)
    chosen=[];history=[];steps=[];best_model=baseline_model if baseline_model is not None else model(bt,yt)
    current=baseline=log_loss(yv,best_model.predict_proba(bv)[:,1],labels=[0,1])
    finished=False
    if (out/'selection_steps.csv').exists():
        steps=pd.read_csv(out/'selection_steps.csv').to_dict('records')
        history=pd.read_csv(out/'selection_trace.csv').to_dict('records')
        np.testing.assert_allclose(steps[0]['validation_log_loss']+steps[0]['gain'],baseline,atol=1e-12,rtol=0)
        chosen=[int(s['candidate']) for s in steps if s['accepted']]
        assert all(j in eligible for j in chosen)
        if chosen:
            with (out/'selection_model.pkl').open('rb') as f:best_model=pickle.load(f)
            current=log_loss(yv,best_model.predict_proba(augment(bv,zv,chosen))[:,1],labels=[0,1])
            saved=[s for s in steps if s['accepted']][-1]['validation_log_loss']
            np.testing.assert_allclose(current,saved,atol=1e-12,rtol=0)
        finished=bool(steps and not steps[-1]['accepted']) or len(chosen)==CFG['feature_budget']
        save(out/'resume_verification.json',dict(previous_steps=len(steps),accepted_indices=chosen,baseline_and_selected_losses_replayed_to_1e_12=True))
        print(out.name,'resumed',len(chosen),'frozen choices',flush=True)
    for step in range(len(chosen),CFG['feature_budget']):
        if finished:break
        remaining=[j for j in eligible if j not in chosen]
        if not remaining:break
        tick=time.perf_counter()
        def evaluate(j):
            st=time.perf_counter();ids=chosen+[j];m=model(augment(bt,zt,ids),yt,initial=best_model)
            loss=log_loss(yv,m.predict_proba(augment(bv,zv,ids))[:,1],labels=[0,1])
            return j,loss,time.perf_counter()-st,m
        scores=[];winning=None;winning_model=None
        with ThreadPoolExecutor(max_workers=CFG['selection_workers']) as executor:
            for j,ll,sec,fitted in executor.map(evaluate,remaining):
                scores.append((j,ll,sec))
                if len(scores)%32==0 or len(scores)==len(remaining):
                    print(out.parent.name,out.name,'step',step+1,'candidates',len(scores),'/',len(remaining),flush=True)
                if winning is None or (ll,j)<winning:
                    winning=(ll,j);winning_model=fitted
        j,loss,_=min(scores,key=lambda r:(r[1],r[0]));gain=current-loss
        for k,ll,sec in scores:history.append(dict(step=step+1,candidate=k,formula=rows[k]['formula_str'],validation_log_loss=ll,gain=current-ll,candidate_count=len(remaining),seconds=sec))
        accepted=gain>CFG['minimum_logloss_gain']
        steps.append(dict(step=step+1,candidate=j,formula=rows[j]['formula_str'],gain=gain,validation_log_loss=loss,candidate_count=len(remaining),seconds=time.perf_counter()-tick,accepted=accepted))
        with (out/'selection_model.pkl').open('wb') as f:pickle.dump(winning_model if accepted else best_model,f,protocol=4)
        pd.DataFrame(history).to_csv(out/'selection_trace.csv',index=False);pd.DataFrame(steps).to_csv(out/'selection_steps.csv',index=False)
        print(out.name,'step',step+1,'gain',gain,'accepted',accepted,flush=True)
        if not accepted:break
        chosen.append(j);current=loss;best_model=winning_model
    final=best_model
    pv=final.predict_proba(augment(bv,zv,chosen))[:,1]
    threshold=max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda t:matthews_corrcoef(yv,pv>=t))
    save(out/'selection.json',dict(indices=chosen,selected=[rows[j] for j in chosen],eligible_count=len(eligible),initial_candidate_count=len(rows),baseline_validation_log_loss=baseline,final_validation_log_loss=current,seconds=time.perf_counter()-start,threshold=threshold,conditioning='A: positional raw + ALL history summary features',stopping='strict positive logloss gain; maximum ten',frozen=True))
    return dict(model=final,formula_scaler=fs,selected=chosen,threshold=threshold,rows=rows)

def fit(dataset):
    out=HERE/dataset;started=time.perf_counter()
    if all((out/f't{t}'/'freeze.json').exists() for t in CFG['landmarks']):
        print(dataset,'all landmarks already frozen; preserving original fit timing',flush=True)
        return
    with (out/'prepared.pkl').open('rb') as f:data=pickle.load(f)
    for parts in data.values():del parts['test']
    props=json.loads((out/'preparation.json').read_text())['props'];rows=json.loads((out/'mining/candidates.json').read_text())
    declare=[dict(formula_tree=t,formula_str=repr(t),family=f,source='DECLARE') for f,t in templates(props)]
    for t in CFG['landmarks']:
        dest=out/f't{t}';dest.mkdir(exist_ok=True)
        if (dest/'frozen.pkl').exists():
            del data[t]
            continue
        d=data[t];tr,va=d['train'],d['val'];yt,yv=tr['label'],va['label']
        if min(np.bincount(yt,minlength=2))<2 or min(np.bincount(yv,minlength=2))<2:
            save(dest/'insufficient.json',dict(reason='training or validation class has fewer than two observations'));continue
        enc=OneHotEncoder(handle_unknown='ignore');rt=enc.fit_transform(tr['raw']);rv=enc.transform(va['raw'])
        scaler=StandardScaler().fit(tr['summary']);bt=hstack([rt,csr_matrix(scaler.transform(tr['summary']))],format='csr');bv=hstack([rv,csr_matrix(scaler.transform(va['summary']))],format='csr')
        # Exact duplicates of existing binary A columns also cannot add information.
        signatures=set();rc=rt.tocsc()
        for j in range(rc.shape[1]):signatures.add(rc.indices[rc.indptr[j]:rc.indptr[j+1]].astype(np.int64).tobytes())
        for col in tr['summary'].T:
            if np.isin(col,[0,1]).all():signatures.add(np.flatnonzero(col).astype(np.int64).tobytes())
        ma=model(bt,yt);pva=ma.predict_proba(bv)[:,1];threshold=max(np.r_[.001,np.arange(.01,1,.01),.999],key=lambda k:matthews_corrcoef(yv,pva>=k))
        fitted={'A':dict(model=ma,threshold=threshold,selected=[],rows=[])}
        for condition,scope,bank in [('B','short',rows),('C','full',rows),('D','full',declare)]:
            target=dest/condition
            if t==10 and condition=='C':
                # Equal input sequences for every session were asserted at preparation.
                ft2=bits(tr['full'],props,[freeze(r['formula_tree']) for r in bank]);fv2=bits(va['full'],props,[freeze(r['formula_tree']) for r in bank])
                assert np.array_equal(ft2,ft) and np.array_equal(fv2,fv)
                fitted['C']=fitted['B'];import shutil;shutil.copytree(dest/'B',target,dirs_exist_ok=True)
                save(target/'t10_identity.json',dict(same_input=True,same_all_candidate_satisfaction=True,reused_identical_selection=True));continue
            trees=[freeze(r['formula_tree']) for r in bank]
            ft=bits(tr[scope],props,trees);fv=bits(va[scope],props,trees)
            fitted[condition]=select(bt,bv,ft,fv,yt,yv,bank,target,signatures,ma)
        bundle=dict(encoder=enc,summary_scaler=scaler,conditions=fitted,props=props)
        for c in ('B','C','D'):
            audit=json.loads((dest/c/'selection.json').read_text())
            np.testing.assert_allclose(audit['baseline_validation_log_loss'],log_loss(yv,pva),atol=1e-12)
        with (dest/'frozen.pkl').open('wb') as f:pickle.dump(bundle,f,protocol=4)
        save(dest/'freeze.json',dict(frozen=True,config_sha256=sha(HERE/'config.json'),model_sha256=sha(dest/'frozen.pkl'),mining_sha256=sha(out/'mining/candidates.json'),test_evaluated=False))
        del data[t]  # Release completed training prefixes before the next landmark.
    save(out/'fit_timing.json',dict(seconds=time.perf_counter()-started))

def evaluate(dataset):
    # No test evaluation until every requested available dataset and landmark is frozen.
    for ds in CFG['datasets']:
        for t in CFG['landmarks']:
            dest=HERE/ds/f't{t}'
            assert (dest/'freeze.json').exists() or (dest/'insufficient.json').exists(),f'Not frozen: {ds}/{t}'
    out=HERE/dataset;started=time.perf_counter()
    with (out/'prepared.pkl').open('rb') as f:data=pickle.load(f)
    for parts in data.values():
        del parts['train'];del parts['val']
    metrics=[];paired=[]
    for t in CFG['landmarks']:
        dest=out/f't{t}'
        if (dest/'insufficient.json').exists():continue
        with (dest/'frozen.pkl').open('rb') as f:bundle=pickle.load(f)
        d=data[t]['test'];y=d['label'];props=bundle['props'];raw=bundle['encoder'].transform(d['raw'])
        base=hstack([raw,csr_matrix(bundle['summary_scaler'].transform(d['summary']))],format='csr')
        probs={};thresholds={};funcs={};point={}
        for c,f in bundle['conditions'].items():
            x=base
            if c!='A':
                scope='short' if c=='B' else 'full'
                fbits=bits(d[scope],props,[freeze(r['formula_tree']) for r in f['rows']])
                if c=='B':btest=fbits
                if c=='C' and t==10:assert np.array_equal(btest,fbits)
                x=augment(base,f['formula_scaler'].transform(fbits),f['selected'])
            p=f['model'].predict_proba(x)[:,1];threshold=f['threshold'];probs[c]=p;thresholds[c]=threshold
            score=score_function(y,p,threshold);err=(y-p)**2
            def scoring(weight,score=score,err=err):return np.r_[score(weight),np.dot(weight,err)/weight.sum()]
            funcs[c]=scoring;point[c]=scoring(np.ones(len(y)))
            np.testing.assert_allclose(point[c],[matthews_corrcoef(y,p>=threshold),average_precision_score(y,p),brier_score_loss(y,p)],atol=1e-12)
        if t==10:np.testing.assert_array_equal(probs['B'],probs['C'])
        pred=pd.DataFrame(dict(session_id=d['session_id'],user_id=d['user_id'],label=y,**probs));pred.to_csv(dest/'predictions.csv.gz',index=False)
        groups=pd.factorize(d['user_id'] if dataset=='Cosmetics' else d['session_id'])[0];ng=int(groups.max())+1
        rng=np.random.default_rng(CFG['bootstrap_seed']);samples={c:[] for c in probs}
        for _ in range(CFG['bootstrap_replicates']):
            weight=np.bincount(rng.integers(ng,size=ng),minlength=ng)[groups]
            for c,fun in funcs.items():samples[c].append(fun(weight))
        samples={c:np.array(v) for c,v in samples.items()}
        for c in probs:
            for j,metric in enumerate(['MCC','AP','Brier']):
                lo,hi=np.nanquantile(samples[c][:,j],[.025,.975])
                metrics.append(dict(dataset=dataset,landmark=t,condition=c,metric=metric,value=point[c][j],lower_95=lo,upper_95=hi,threshold=thresholds[c],n_test=len(y),positives=int(y.sum()),features=len(bundle['conditions'][c]['selected'])))
        for comparator in ['A','B','D']:
            for j,metric in enumerate(['MCC','AP','Brier']):
                values=samples['C'][:,j]-samples[comparator][:,j];lo,hi=np.nanquantile(values,[.025,.975])
                paired.append(dict(dataset=dataset,landmark=t,contrast='C-'+comparator,metric=metric,delta=point['C'][j]-point[comparator][j],lower_95=lo,upper_95=hi,unit='user' if dataset=='Cosmetics' else 'session',units=ng,replicates=len(values),valid_replicates=int(np.isfinite(values).sum())))
        save(dest/'evaluation_audit.json',dict(shared_rows_labels=True,cohort_sha256=sha(out/f'cohort_t{t}_test.csv.gz'),prediction_sha256=sha(dest/'predictions.csv.gz'),t10_B_C_identical=t==10,model_and_formula_fixed=True))
        print(dataset,t,'test evaluation done',flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    save(out/'evaluation_timing.json',dict(seconds=time.perf_counter()-started))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        {'fit':fit,'evaluate':evaluate}[sys.argv[1]](sys.argv[2])
