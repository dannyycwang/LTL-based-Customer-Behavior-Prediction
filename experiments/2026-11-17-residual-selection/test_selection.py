"""Synthetic algorithm check; not an experiment or a paper result."""
import tempfile
from pathlib import Path
import numpy as np
from run_residual import select

def main():
    rng=np.random.default_rng(17)
    yt=np.tile([0,1],100)
    yv=np.tile([0,1],50)
    ft=rng.integers(2,size=(200,12))
    fv=rng.integers(2,size=(100,12))
    ft[:,10]=yt;fv[:,10]=yv
    ft[:,11]=yt;fv[:,11]=yv
    rows=[dict(source='candidate_export',formula_str=f'synthetic_{i}') for i in range(12)]
    with tempfile.TemporaryDirectory() as folder:
        selected=select(np.zeros((200,1)),np.zeros((100,1)),ft,fv,yt,yv,rows,Path(folder))
    assert selected[0]==10,selected
    assert 11 not in selected, 'Duplicate signal must not change effective regularization'
    assert len(selected)==len(set(selected))==10
    print('PASS: conditional signal recovered, duplicate tie resolved deterministically, K=10')

if __name__=='__main__': main()
