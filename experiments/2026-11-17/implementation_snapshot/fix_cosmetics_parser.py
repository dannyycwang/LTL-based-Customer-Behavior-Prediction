"""Parse the prepared Cosmetics event encoding while preserving event positions.

For the recorded prop_BP format, whitespace separates EVENTS and '+' separates
propositions within one event. Python-literal list traces are also supported.
This module never modifies an existing dataset or Z file.
"""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from diagnose_coveo import vector
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]/'work'

def parse_event_trace(value):
    if isinstance(value,(list,tuple)):
        states=value
    else:
        if not isinstance(value,str) or not value.strip():raise ValueError('Empty/invalid trace')
        try:parsed=ast.literal_eval(value)
        except (ValueError,SyntaxError):parsed=None
        if isinstance(parsed,(list,tuple)):states=parsed
        elif ';' in value:states=value.split(';')
        else:states=value.split()
    result=[]
    for s in states:
        if isinstance(s,(set,list,tuple)):result.append({str(a) for a in s})
        else:result.append(set(str(s).replace('+',' ').replace('|',' ').split()))
    return result

def main():
    df=pd.read_csv(WORK/'cosmetics_seed42_test.csv',dtype={'session_id_hash':str})
    nb=json.loads((WORK/'electronics_nb_0.ipynb').read_text(encoding='utf-8'))
    src=''.join(nb['cells'][18]['source']).split('\n',1)[1]
    mod=ast.parse(src);names={'safe_literal_eval','normalize_prop','split_state_string','parse_prop_trace_cell','eval_formula_tree'}
    scope={'ast':ast,'pd':pd,'np':np}
    exec(compile(ast.Module(body=[n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'Dataset3.ipynb:cell18','exec'),scope)
    basis=json.loads((WORK/'cosmetics_BP_basis_clean.json').read_text())
    next_formula=next(ast.literal_eval(r['formula_tree']) for r in basis['formulas'][:10] if r['formula_str']=='X(hp)')
    rows=[]
    for i,r in df.iterrows():
        old=scope['parse_prop_trace_cell'](r.prop_BP);new=parse_event_trace(r.prop_BP)
        assert len(new)==int(r.n_events)
        if i==0:
            rows.append(dict(session_id=r.session_id_hash,n_events=int(r.n_events),old_parsed_states=len(old),new_parsed_states=len(new),formula='X(hp)',old_value=int(scope['eval_formula_tree'](next_formula,old)),new_value=int(vector(new,[next_formula])[0]),raw_prop_BP=r.prop_BP))
        assert new==[set(t.split('+')) for t in r.prop_BP.split()]
    pd.DataFrame(rows).to_csv(ROOT/'cosmetics_parser_real_example.csv',index=False)
    print('Verified corrected parser on all',len(df),'test sessions. Example:',rows[0])

if __name__=='__main__':main()
