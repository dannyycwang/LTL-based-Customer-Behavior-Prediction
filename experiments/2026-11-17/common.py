import ast, json, os, sys, pickle
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
AUDIT=PROJECT/'outputs/research_audit'
sys.path.insert(0,str(PROJECT/'work/python_packages'))
sys.path.insert(0,str(AUDIT))
import numpy as np
import pandas as pd
from diagnose_coveo import freeze,vector,sha
def read_basis(path):
    rows=json.loads(Path(path).read_text(encoding='utf-8'))
    return (rows['formulas'] if isinstance(rows,dict) else rows)
def tree(row):
    t=row['formula_tree']; return freeze(ast.literal_eval(t) if isinstance(t,str) else t)
def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def load_data(dataset):
    with (HERE/'prepared'/f'{dataset}.pkl').open('rb') as f:return pickle.load(f)
