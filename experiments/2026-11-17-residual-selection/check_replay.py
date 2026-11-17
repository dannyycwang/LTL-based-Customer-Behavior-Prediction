"""Verify A/B reproduce the frozen historical control predictions."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

old=Path('C:/Users/ycw/Documents/Codex/2026-11-10/new-chat/outputs/research_audit/submission_iterations_seed42')
new=Path(sys.argv[1])
rows=[]
for file in new.glob('*/*_predictions.csv.gz'):
    previous=old/file.name
    a=pd.read_csv(file,dtype={'session_id':str})
    b=pd.read_csv(previous,dtype={'session_id':str})
    assert np.array_equal(a.session_id,b.session_id)
    assert np.array_equal(a.label,b.label)
    for current,historical in [('A','Short + Counts + transitions'),('B','Short + Counts + transitions + LTLf')]:
        error=float(np.max(np.abs(a[current]-b[historical])))
        rows.append(dict(file=file.name,condition=current,max_probability_difference=error))
        # Report numerical replay deviations; never tune to test performance.
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(new/'historical_replay.csv',index=False)
