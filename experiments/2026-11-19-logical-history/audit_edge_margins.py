"""Read-only extraction-edge audit; no alternative cohort or model is evaluated."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,str(Path(CFG['source_project'])/'work/python_packages'))
import pandas as pd
meta=json.loads((HERE/'Coveo/raw_audit.json').read_text())
raw=pd.read_pickle(HERE/'Coveo/raw_audit.pkl')
gap=meta['max_within_session_gap_ms']/60000
rows=[];flagged=set()
for t in CFG['landmarks']:
    for split in ('train','val','test'):
        cohort=pd.read_csv(HERE/f'Coveo/cohort_t{t}_{split}.csv.gz',dtype={'session_id':str})
        r=raw.loc[cohort.session_id]
        left=(r.start_ms-meta['source_min_ms'])/60000
        unresolved=r.first_purchase.isna() & (r.total_events<t+CFG['horizon'])
        right=(meta['source_max_ms']-r.loc[unresolved,'end_ms'])/60000
        flagged.update(right[right<=gap].index)
        rows.append(dict(landmark=t,split=split,min_retained_left_margin_minutes=float(left.min()),
            min_unresolved_right_margin_minutes=float(right.min()) if len(right) else None,
            retained_left_within_max_observed_gap=int((left<=gap).sum()),
            unresolved_right_within_max_observed_gap=int((right<=gap).sum()),
            retained_left_within_60_minutes=int((left<=60).sum()),
            unresolved_right_within_60_minutes=int((right<=60).sum())))
out=dict(official_guard_minutes=30,maximum_observed_within_session_gap_minutes=gap,cohorts=rows,
    scope='Read-only edge margin check; does not redefine the fixed 30-minute policy or run alternative models')
mined=pd.read_csv(HERE/'Coveo/mining/mining_prefix_ids.csv.gz',dtype={'session_id':str})
out['flagged_session_count']=len(flagged)
out['flagged_sessions_in_mining_pool']=int(mined.session_id.isin(flagged).sum())
(HERE/'coveo_edge_margin_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
