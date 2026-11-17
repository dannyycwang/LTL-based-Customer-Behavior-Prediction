"""Describe the preselected thresholds; never choose a replacement from test."""
from common import *
from sklearn.metrics import matthews_corrcoef
rows=[]
for path in (HERE/'neural').glob('Electronics_ltlf_*/metrics.json'):
    for split,name in [('validation','validation_predictions.csv.gz'),('test','predictions.csv.gz')]:
        d=pd.read_csv(path.parent/name)
        rows.append(dict(run=path.parent.name,split=split,n=len(d),threshold=float(d.threshold.iloc[0]),predicted_positive=int((d.probability>=d.threshold).sum()),MCC=matthews_corrcoef(d.label,d.probability>=d.threshold)))
pd.DataFrame(rows).to_csv(HERE/'audit/electronics_decision_diagnostics.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
