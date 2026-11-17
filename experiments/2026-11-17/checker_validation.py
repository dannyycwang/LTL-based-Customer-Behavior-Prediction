from common import *
from diagnose_coveo import load_reference,REFERENCE
from formula_audit import atoms
from finite_equivalence import lit,step,accept,equivalent
import itertools
def main():
    src=Path('C:/Users/ycw/TLTLf/interpret.py');module=ast.parse(src.read_text(encoding='utf-8'));node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='LTLf')
    scope={'sys':sys};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),scope);cls=scope['LTLf'];ref=load_reference(REFERENCE)
    tests=['true','false',('X','true'),('X',('!','a')),('U','a','b'),('F','a'),('G','a'),('&','a','b')]
    for ds in ['Coveo','Electronics','Cosmetics']:
        meta=json.loads((HERE/'prepared'/f'{ds}_manifest.json').read_text());tests.extend(tree(r) for r in meta['formulas'][:10])
    rng=np.random.default_rng(421116);checks=0
    for t in tests:
        vocab=sorted(atoms(t))
        for n in [1,2,3,5,10,20]:
            for k in range(50):
                tr=[{a for a in vocab if rng.random()<.4} for _ in range(n)]
                actual=vector(tr,[t])[0];expected=cls(vocab,t).pathCheck([list(s) for s in tr],str(k))
                assert actual==expected==ref(t,tr,0,{})
                checks+=1
    # Every reported non-equivalence has a replayable finite counterexample.
    pairs=pd.read_csv(HERE/'audit/formula_pairs.csv');witnesses=0
    for ds,g in pairs.groupby('dataset'):
        records=json.loads((HERE/'prepared'/f'{ds}_manifest.json').read_text())['formulas']
        for r in g.itertuples():
            if r.logic_status=='not_equivalent':
                tr=[set(s) for s in json.loads(r.witness)];bits=vector(tr,[tree(records[r.first-1]),tree(records[r.second-1])]);assert bits[0]!=bits[1];witnesses+=1
    save_json(HERE/'audit/checker_validation.json',dict(status='passed',three_evaluator_comparisons=checks,finite_counterexamples_replayed=witnesses,original_checker_sha256=sha(src),bit_checker_sha256=sha(AUDIT/'diagnose_coveo.py'),reference_sha256=sha(REFERENCE),strong_next_final=False,until_final='right operand only',multi_proposition=True))
if __name__=='__main__':main()
