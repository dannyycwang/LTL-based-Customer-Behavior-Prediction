"""Record inherited negative-price filtering; no model/cohort mutation."""
import csv,gzip,json,sys,time
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
from collections import Counter
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
WORK=Path(CFG['source_project'])/'work'
csv.field_size_limit(2**31-1)
start=time.perf_counter()
eligible=set()
for path in (HERE/'Cosmetics').glob('cohort_t*_*.csv.gz'):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        eligible.update(r['session_id'] for r in csv.DictReader(f))
keys={}
for split in ('train','val','test'):
    with (WORK/f'cosmetics_seed42_{split}.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            keys[(r['user_id'],r['user_session'])]=r['session_id_hash']
counts=Counter();affected=set();eligible_affected=set();negative_purchases=set()
with ZipFile(WORK/'cosmetics_dec_source_download.zip') as z,z.open('2019-Dec.csv') as b:
    for r in csv.DictReader(TextIOWrapper(b,encoding='utf-8',newline='')):
        try:price=float(r['price'])
        except (ValueError,TypeError):continue
        if not price<0:continue
        counts['all_source_negative_price_events']+=1
        key=(r['user_id'],r['user_session'])
        if key not in keys:continue
        sid=keys[key];affected.add(sid);counts['inherited_cohort_'+r['event_type']]+=1
        if sid in eligible:
            eligible_affected.add(sid);counts['eligible_union_'+r['event_type']]+=1
            if r['event_type']=='purchase':negative_purchases.add(sid)
out=dict(counts=counts,inherited_sessions_with_negative_events=len(affected),
    eligible_union_sessions=len(eligible),eligible_union_sessions_with_negative_events=len(eligible_affected),
    eligible_union_sessions_with_negative_purchase_events=len(negative_purchases),
    seconds=time.perf_counter()-start,policy='Existing pipeline excludes negative-price rows before event indexing; no mutation by this audit')
target_keys={k:s for k,s in keys.items() if s in eligible_affected}
events={s:[] for s in eligible_affected}
with ZipFile(WORK/'cosmetics_dec_source_download.zip') as z,z.open('2019-Dec.csv') as b:
    for i,r in enumerate(csv.DictReader(TextIOWrapper(b,encoding='utf-8',newline=''))):
        key=(r['user_id'],r['user_session'])
        if key not in target_keys:continue
        try:price=float(r['price'])
        except (ValueError,TypeError):price=0.
        events[target_keys[key]].append((r['event_time'],i,r['event_type'],bool(price<0)))
checks=[]
for sid,rows in events.items():
    rows.sort();clean=[r for r in rows if not r[3]]
    first_raw=next((i+1 for i,r in enumerate(rows) if r[2]=='purchase'),None)
    first_clean=next((i+1 for i,r in enumerate(clean) if r[2]=='purchase'),None)
    before=sum(r[3] for r in rows[:first_raw]) if first_raw is not None else sum(r[3] for r in rows)
    same_prefix=(rows[:first_raw-1]==clean[:first_clean-1]) if first_raw is not None and first_clean is not None else rows==clean
    checks.append(dict(first_purchase_position_raw=first_raw,first_purchase_position_clean=first_clean,
        negative_events_at_or_before_first_raw_purchase=before,
        all_pre_purchase_events_and_first_purchase_index_unchanged=same_prefix and first_raw==first_clean))
out['affected_session_checks_without_identifiers']=checks
out['all_eligible_prefixes_and_future_first_purchase_labels_unchanged_by_cleaning']=all(r['all_pre_purchase_events_and_first_purchase_index_unchanged'] for r in checks)
out['seconds']=time.perf_counter()-start
(HERE/'price_cleaning_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
assert out['all_eligible_prefixes_and_future_first_purchase_labels_unchanged_by_cleaning'],'Cleaning changes the requested first-purchase task'
