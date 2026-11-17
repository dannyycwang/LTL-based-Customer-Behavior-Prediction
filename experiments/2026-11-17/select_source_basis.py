"""Replay the actual notebook clean_pack rule, before source-control test evaluation."""
from common import *
def select():
    out=HERE/'cosmetics_source';source=out/'BP/BP_all_candidates.csv'
    marker=out/'final_selection.json'
    if marker.exists():
        assert json.loads(marker.read_text())['basis_sha256']==sha(out/'basis.json');return
    d=pd.read_csv(source);vocab=json.loads((PROJECT/'work/cosmetics_BP_basis_clean.json').read_text())['vocab']
    def leaves(t):return {t} if isinstance(t,str) else set().union(*(leaves(c) for c in t[1:]))
    keep=d.formula_tree.map(lambda t:bool(leaves(freeze(ast.literal_eval(t)))&set(vocab)))
    clean=d.loc[keep].sort_values(['balanced_accuracy','support_gap','pos_support'],ascending=[False,False,False]).drop_duplicates('formula_str').head(20).copy()
    clean['formula_id']=[f'T{i:02}' for i in range(1,len(clean)+1)]
    assert len(clean)>=10
    if (out/'basis.json').exists():(out/'basis.json').rename(out/'pre_cleanup_untested_basis.json')
    save_json(out/'basis.json',clean.head(10).to_dict('records'));clean.to_csv(out/'clean_top20.csv',index=False)
    save_json(marker,dict(basis_sha256=sha(out/'basis.json'),source_sha256=sha(source),rule='Actual clean_pack from electronics_nb_0_cells.txt lines 744-776: all candidates preferred; remove formulas with no dataset atoms; sort balanced_accuracy, signed support_gap, pos_support descending; drop duplicate formula_str; top20 then first10',not_empirical_dedup=True,reason='Recovered notebook cleanup overrides misleading semantic_unique file naming. This rule was identified before any new basis-source test evaluation.',primary_basis_unchanged=True))
if __name__=='__main__':select()
