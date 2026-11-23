"""Read-only boundary follow-up using the separate original search activity stream."""
import sys,json,time
from pathlib import Path
from zipfile import ZipFile
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
sys.path.insert(0,str(Path(CFG['source_project'])/'work/python_packages'))
import pandas as pd
meta=json.loads((HERE/'Coveo/raw_audit.json').read_text());raw=pd.read_pickle(HERE/'Coveo/raw_audit.pkl')
cohorts=[];ids=set();short_ids=set()
for t in CFG['landmarks']:
    for split in ('train','val','test'):
        cohort=pd.read_csv(HERE/f'Coveo/cohort_t{t}_{split}.csv.gz',dtype={'session_id':str})
        ids.update(cohort.session_id)
        short_ids.update(cohort.loc[cohort.natural_short_negative==1,'session_id'])
        cohorts.append((t,split,cohort))
r=raw.loc[list(short_ids)]
margin=(meta['source_max_ms']-r.end_ms)/60000
flagged=set(r.index[margin<=meta['max_within_session_gap_ms']/60000])
latest={};earliest={};events={};source_max=0;source_min=float('inf');start=time.perf_counter()
with ZipFile(CFG['coveo_raw']) as z,z.open('train/search_train.csv') as stream:
    for chunk in pd.read_csv(stream,usecols=['session_id_hash','server_timestamp_epoch_ms'],dtype={'session_id_hash':str},chunksize=100000):
        ts=pd.to_numeric(chunk.server_timestamp_epoch_ms,errors='coerce')
        source_max=max(source_max,float(ts.max()));source_min=min(source_min,float(ts.min()))
        hit=chunk[chunk.session_id_hash.isin(ids)]
        for sid,part in hit.groupby('session_id_hash'):
            values=pd.to_numeric(part.server_timestamp_epoch_ms,errors='coerce').dropna()
            latest[sid]=max(latest.get(sid,0),float(values.max()))
            earliest[sid]=min(earliest.get(sid,float('inf')),float(values.min()))
            events[sid]=events.get(sid,0)+len(values)
end=max(source_max,meta['source_max_ms']);begin=min(source_min,meta['source_min_ms']);checks=[]
left_flags=set();right_flags=set();min_right=float('inf');min_left=float('inf')
for sid in sorted(ids):
    browse=float(raw.loc[sid,'end_ms']);last=max(browse,latest.get(sid,0))
    first=min(float(raw.loc[sid,'start_ms']),earliest.get(sid,float('inf')))
    min_left=min(min_left,(first-begin)/60000)
    if first-begin<=1800000:left_flags.add(sid)
    if sid in short_ids:
        min_right=min(min_right,(end-last)/60000)
        if end-last<1800000:right_flags.add(sid)
    if sid in flagged:checks.append(dict(browsing_events=int(raw.loc[sid,'total_events']),search_events=events.get(sid,0),
        browsing_end_margin_minutes=(end-browse)/60000,
        last_any_observed_activity_margin_minutes=(end-last)/60000,
        closed_before_capture_end_under_official_30_minute_rule=end-last>=1800000))
summary=[]
for t,split,cohort in cohorts:
    summary.append(dict(landmark=t,split=split,left_activity_boundary_flags=int(cohort.session_id.isin(left_flags).sum()),
        right_unresolved_activity_boundary_flags=int(cohort.session_id.isin(right_flags).sum())))
out=dict(search_source_member='train/search_train.csv',search_min_ms=source_min,search_max_ms=source_max,
    checks_without_identifiers=checks,seconds=time.perf_counter()-start,
    retained_session_count=len(ids),natural_short_negative_session_count=len(short_ids),
    left_activity_boundary_flag_count=len(left_flags),right_unresolved_activity_boundary_flag_count=len(right_flags),
    minimum_retained_left_any_activity_margin_minutes=min_left,minimum_short_negative_right_any_activity_margin_minutes=min_right,
    cohort_flags=summary,all_retained_sessions_pass_combined_activity_boundary_check=not(left_flags or right_flags),
    all_flagged_sessions_have_30_minutes_of_followup_after_last_any_activity=all(c['closed_before_capture_end_under_official_30_minute_rule'] for c in checks),
    scope='Search events used only to audit closure; no change to browsing-event input/target task or candidate pool')
(HERE/'coveo_search_closure_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
assert out['all_retained_sessions_pass_combined_activity_boundary_check'],'Combined activity reveals unresolved extraction boundaries'
