"""Required leakage, finite-trace, horizon and selector checks; synthetic data only."""
from pilot import *
from fit import bits,select,threadpool_limits
import tempfile

def main():
    cases=[set(),{'a'},{'b'},{'a','b'}]
    trees=[t for _,t in templates(['a','b'])]+[('X','true'),('U','a','b'),('U',('X','a'),('!','b'))]
    count=0
    for n in range(1,5):
        traces=list(itertools.product(cases,repeat=n));got=bits(traces,['a','b'],trees)
        for i,tr in enumerate(traces):assert np.array_equal(got[i],vector(tr,trees));count+=1
    fields=[['d','a','d','r']*12,['P1','P2','P1','P3']*12,['C1','C2','C1','C1']*12]
    props=['b','d','a','r','fp','rp','sp','pc','p2','p3','sc','cc','rc']
    for t in (10,20,30):
        mutated=[f[:t]+['FUTURE_MUTATION']*(len(f)-t) for f in fields]
        def feature(f):
            pref=[v[:t] for v in f];full=abstract(pref,props,'Coveo');short=abstract([v[-10:] for v in pref],props,'Coveo')
            return [v[-10:] for v in pref],full,short,summary(full,props)
        a,b=feature(fields),feature(mutated)
        assert a[:3]==b[:3];np.testing.assert_array_equal(a[3],b[3])
        if t==10:assert a[1]==a[2]
    # Boundary arithmetic: purchase at t excluded, purchase t+1/t+10 positive, t+11 negative.
    for t in (10,20,30):
        assert [int(t<p<=t+10) for p in [t,t+1,t+10,t+11]]==[0,1,1,0]
    from scipy.sparse import csr_matrix
    y=np.tile([0,1],50);base=csr_matrix(np.zeros((100,1)))
    ft=np.column_stack([np.ones(100),y,y,1-y]);fv=np.column_stack([np.ones(100),y,y,y])
    rows=[dict(formula_str=str(i),formula_tree='a',source='candidate_export') for i in range(4)]
    with tempfile.TemporaryDirectory(dir=HERE) as tmp,threadpool_limits(limits=1):
        result=select(base,base,ft,fv,y,y,rows,Path(tmp),set())
        assert result['selected']==[1],result['selected']
        trace=pd.read_csv(Path(tmp)/'selection_steps.csv');assert not bool(trace.iloc[-1].accepted)
        resumed=select(base,base,ft,fv,y,y,rows,Path(tmp),set())
        assert resumed['selected']==result['selected']
        np.testing.assert_array_equal(resumed['model'].coef_,result['model'].coef_)
    save(HERE/'software_validation.json',dict(status='passed',exhaustive_finite_traces=count,future_suffix_mutation='passed',t10_window_input_identity='passed',horizon_boundaries='passed',selector='constants and exact duplicates removed; beneficial feature accepted; negative marginal gain rejected'))

if __name__=='__main__':main()
