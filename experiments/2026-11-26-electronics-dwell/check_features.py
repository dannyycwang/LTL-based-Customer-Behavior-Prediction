"""Independent arithmetic checks; no classifier fit and no test predictions."""
from common import *
from prepare import make,gaps,audit_unit
from collections import Counter

def main():
    audit_unit();checked=0
    for ds in CFG['datasets']:
        d=load(privdir(ds)/'prepared.pkl')['train'];meta=json.loads((outdir(ds)/'preparation.json').read_text())
        ix=np.random.default_rng(42).choice(len(d['label']),min(100,len(d['label'])),replace=False)
        for i in ix:
            tr=d['full'][i];g=d['gaps'][i];v,ext,names=make(tr,g,meta['props'],meta['buckets']);got=dict(zip(names,v))
            np.testing.assert_array_equal(v,d['time'][i]);assert ext==d['dwell'][i]
            for a in ['prefix','action_b','action_d','action_a','action_r']:
                valid=[float(x) for j,x in enumerate(g) if np.isfinite(x) and (a=='prefix' or a[-1] in tr[j])]
                assert got[a+'_known_n']==len(valid)
                np.testing.assert_allclose(got[a+'_sum_sec'],sum(valid))
                np.testing.assert_allclose(got[a+'_mean_sec'],sum(valid)/len(valid) if valid else 0)
                np.testing.assert_allclose(got[a+'_std_sec'],(sum((x-sum(valid)/len(valid))**2 for x in valid)/len(valid))**.5 if valid else 0)
            pair=Counter((a,b) for left,right in zip(ext,ext[1:]) for a in left for b in right if a in DWELL or b in DWELL)
            for a in meta['props']+DWELL:
                for b in meta['props']+DWELL:
                    if a in DWELL or b in DWELL:
                        assert got[f'pair_{a}_{b}_occurrence']==int(pair[a,b]>0)
                        assert got[f'pair_{a}_{b}_frequency']==pair[a,b]/19
            # Appending arbitrary future propositions cannot change any input.
            altered=(tr+[{'p','dh','fake_future_product'}]*10)[:20]
            vv,ee,_=make(altered,g,meta['props'],meta['buckets']);np.testing.assert_array_equal(v,vv);assert ext==ee
            checked+=1
        known=d['gaps'][np.isfinite(d['gaps'])];cap=np.quantile(known,.99)
        assert cap==meta['buckets']['cap_seconds']
        np.testing.assert_array_equal(np.quantile(np.log1p(np.minimum(known,cap)),[1/3,2/3]),meta['buckets']['q_log'])
    save(HERE/'independent_feature_audit.json',dict(passed=True,real_training_prefixes=checked,checks=['independent statistics','independent Cartesian adjacent pairs','future event invariance','train-only bucket reconstruction','exact saved T/logic replay'],classifier_fits=0,test_predictions=0))
    print('Independent feature audit passed:',checked,flush=True)
if __name__=='__main__':main()
