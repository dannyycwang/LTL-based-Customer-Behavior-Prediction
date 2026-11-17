from common import *
from linear import fit
from fix_cosmetics_parser import parse_event_trace
from bootstrap_submission_controls import score_function
from threadpoolctl import threadpool_limits

def main():
    from select_source_basis import select
    select()
    out=HERE/'cosmetics_source/comparison';out.mkdir(parents=True,exist_ok=True)
    basis=HERE/'cosmetics_source/basis.json';trees=[tree(r) for r in read_basis(basis)];assert len(trees)==10
    data=load_data('Cosmetics');zs={}
    for s,d in data.items():
        frame=pd.read_csv(PROJECT/'work/cosmetics_pre_purchase_seed42'/f'{s}.csv',usecols=['session_id_hash','prop_BP'],dtype={'session_id_hash':str}).set_index('session_id_hash')
        bits=np.array([vector(parse_event_trace(v)[:10],trees) for v in frame.loc[d['session_id'],'prop_BP']])
        zs[s]={'frozen_historical':np.column_stack([d['counts'],d['ltlf']]),'relearned_pre_purchase':np.column_stack([d['counts'],bits])}
    save_json(out/'basis_manifest.json',dict(new_basis_sha256=sha(basis),primary_basis_unchanged=True,downstream_scope='event10 H10',basis_source_scope='last20 before first purchase',labels='eventual purchase for both formula learners; next10 only downstream'))
    fit('Cosmetics',data,zs,out,'basis_source_control')
    a=pd.read_csv(out/'relearned_pre_purchase/predictions.csv.gz',dtype={'session_id':str,'user_id':str});b=pd.read_csv(out/'frozen_historical/predictions.csv.gz',dtype={'session_id':str,'user_id':str})
    assert np.array_equal(a.session_id,b.session_id)
    groups=pd.factorize(a.user_id)[0];n=groups.max()+1;rng=np.random.default_rng(20261116)
    fa=score_function(a.label.to_numpy(),a.probability.to_numpy(),a.threshold.iloc[0]);fb=score_function(b.label.to_numpy(),b.probability.to_numpy(),b.threshold.iloc[0])
    reps=[]
    for _ in range(1500):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[groups];reps.append(fa(w)-fb(w))
    reps=np.array(reps);np.save(out/'paired_difference_replicates.npy',reps);lo,hi=np.quantile(reps,[.025,.975],axis=0);point=fa(np.ones(len(a),int))-fb(np.ones(len(a),int))
    pd.DataFrame([dict(metric=m,delta=point[i],lower95=lo[i],upper95=hi[i],contrast='pre-purchase minus frozen historical',bootstrap_unit='user',model_seed=42) for i,m in enumerate(['MCC','AP'])]).to_csv(out/'paired_differences.csv',index=False)
    pd.DataFrame([json.loads((out/c/'metrics.json').read_text()) for c in zs['train']]).to_csv(out/'metrics.csv',index=False)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
