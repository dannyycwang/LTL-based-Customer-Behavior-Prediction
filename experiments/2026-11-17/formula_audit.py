"""Conservative finite-trace simplification; empirical identity is separately reported."""
from common import *
import itertools
from finite_equivalence import equivalent,lit,step,accept

def simplify(t):
    if isinstance(t,str):return t
    op={'&&':'&','||':'|'}.get(t[0],t[0]);a=simplify(t[1]);b=simplify(t[2]) if len(t)>2 else None
    if op=='U' and a=='true':return simplify(('F',b))
    if op in ['F','G'] and isinstance(a,tuple) and a[0]==op:return a
    if op=='&':
        if a=='true':return b
        if b=='true' or a==b:return a
    if op=='U' and a==b:return a
    return (op,a,b) if b is not None else (op,a)
def atoms(t):return {t}-{ 'true','false'} if isinstance(t,str) else set().union(*(atoms(c) for c in t[1:]))
def size(t):return 1 if isinstance(t,str) else 1+sum(size(c) for c in t[1:])
def ops(t):return set() if isinstance(t,str) else {t[0]}|set().union(*(ops(c) for c in t[1:]))
def fmt(t):
    if isinstance(t,str):return t
    return f'{t[0]}({fmt(t[1])})' if len(t)==2 else f'({fmt(t[1])} {t[0]} {fmt(t[2])})'
def meaning(t):
    if isinstance(t,str):return t+' holds at the current event'
    op=t[0]
    if op=='X':return 'There is a next event at which '+meaning(t[1])
    if op=='F':return 'At some current or later event, '+meaning(t[1])
    if op=='U':return 'At some current or later event '+fmt(t[2])+' holds, and '+fmt(t[1])+' holds at every earlier event'
    if op=='&':return meaning(t[1])+'; also '+meaning(t[2])
    return fmt(t)

def main():
    out=HERE/'audit';out.mkdir(exist_ok=True);rows=[];pairs=[];proofs=[]
    for ds in ['Coveo','Electronics','Cosmetics']:
        meta=json.loads((HERE/'prepared'/f'{ds}_manifest.json').read_text());records=meta['formulas'][:10];data=load_data(ds)
        trees=[tree(r) for r in records];simple=[simplify(t) for t in trees]
        for i,t in enumerate(simple):
            candidates=['true','false']
            for atom in sorted(atoms(t)):
                x=atom
                for k in range(7):
                    candidates.extend([x,('F',x)]);x=('X',x)
            for c in sorted(candidates,key=size):
                if size(c)>=size(simple[i]):continue
                result=equivalent(t,c,cap=2000)
                if result['status']=='equivalent':
                    simple[i]=c;proofs.append(dict(dataset=ds,index=i+1,original=fmt(t),simplified=fmt(c),**result));break
        for i,(r,t,s) in enumerate(zip(records,trees,simple)):
            # Random/boundary validation supplements, but does not prove, the named rewrite rules.
            rng=np.random.default_rng(42);v=sorted(atoms(t));tested=0
            for n in [1,2,3,4,5,10,20]:
                for _ in range(100):
                    tr=[{a for a in v if rng.random()<.5} for _ in range(n)]
                    assert vector(tr,[t,s])[0]==vector(tr,[t,s])[1];tested+=1
                    state=lit(t)
                    for event in tr:state=step(state,frozenset(event))
                    assert accept(state)==bool(vector(tr,[t])[0])
            temporal=sorted(ops(s)&{'X','F','G','U'})
            rows.append(dict(dataset=ds,index=i+1,original=r['formula_str'],simplified=fmt(s),
                simplification_status='finite-trace identities and exact residual-language equivalence; no proposition invariants assumed',meaning=meaning(s),
                atoms=','.join(sorted(atoms(t))),ast_nodes=size(t),simplified_nodes=size(s),temporal_operators=','.join(temporal),
                pattern='occurrence' if isinstance(s,tuple) and s[0]=='F' and isinstance(s[1],str) else 'next/occurrence' if 'U' not in temporal else 'until with event/state guards',
                checks=tested,original_ranking_positive_rate=r.get('pos_support'),original_ranking_negative_rate=r.get('neg_support'),basis_sha256=meta['basis_sha256']))
        for i in range(10):
            for j in range(i):
                train_equal=np.array_equal(data['train']['ltlf'][:,i],data['train']['ltlf'][:,j]);val_equal=np.array_equal(data['val']['ltlf'][:,i],data['val']['ltlf'][:,j])
                proof=equivalent(simple[i],simple[j],cap=2000)
                pairs.append(dict(dataset=ds,first=i+1,second=j+1,train_columns_equal=train_equal,val_columns_equal=val_equal,logic_status=proof['status'],witness=json.dumps(proof.get('witness')),automaton_states=proof['states']))
    pd.DataFrame(rows).to_csv(out/'all_formulas.csv',index=False);pd.DataFrame(pairs).to_csv(out/'formula_pairs.csv',index=False)
    save_json(out/'simplification_proofs.json',proofs)
    pd.concat([pd.read_csv(HERE/'prepared'/f'{d}_formula_support.csv') for d in ['Coveo','Electronics','Cosmetics']]).to_csv(out/'formula_support.csv',index=False)
    save_json(out/'semantic_sources.json',{'finite_traces':'https://www.cs.rice.edu/~vardi/papers/ijcai13.pdf','DECLARE_encoding':'https://ceur-ws.org/Vol-3310/paper9.pdf','rules':['true U p = F p','F F p = F p','G G p = G p','p AND true = p','p U p = p'],'logical_equivalence_scope':'Exact residual-language BFS for positive Boolean/X/U/F/G formulas on nonempty finite traces. A state cap produces unknown. Counterexamples saved for inequivalent pairs. Residual evaluator cross-checked against existing checker on 21000 random/boundary traces.','multi_proposition':'Unrestricted subsets, including empty states and simultaneous atoms. No exclusive-action assumption in semantic tests.','template_precedence':'G(!b) OR (!b U (a AND !b)); strict earlier a required for first b. Unlike event-exclusive syntax, simultaneous a,b does not count as precedence.'})
if __name__=='__main__':main()
