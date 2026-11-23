"""Download original source months for boundary auditing only, no new model cohort."""
from pilot import *
import requests
from concurrent.futures import ThreadPoolExecutor

def main():
    dest=HERE/'private_boundary_sources';dest.mkdir(exist_ok=True)
    months=['2019-Oct','2019-Nov','2020-Jan','2020-Feb']
    url='https://www.kaggle.com/api/v1/datasets/download/mkechinov/ecommerce-events-history-in-cosmetics-shop/'
    def get(month):
        path=dest/(month+'.zip')
        if not path.exists():
            temp=path.with_suffix('.part')
            with requests.get(url+month+'.csv',stream=True,timeout=180) as r:
                r.raise_for_status()
                with temp.open('wb') as f:
                    for block in r.iter_content(1048576):f.write(block)
            with ZipFile(temp) as z:assert month+'.csv' in z.namelist()
            temp.rename(path)
        print('source available',month,path.stat().st_size,flush=True)
        return path
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(get,months))
    maps=pd.concat([pd.read_csv(WORK/f'cosmetics_seed42_{s}.csv',usecols=['user_id','user_session','session_id_hash'],dtype=str) for s in SPLITS])
    sessions=set(maps.user_session);matches=[];metadata=[]
    for month,path in zip(months,paths):
        count=0
        with ZipFile(path) as z,z.open(month+'.csv') as stream:
            for chunk in pd.read_csv(stream,usecols=['user_id','user_session','event_time'],dtype=str,chunksize=300000):
                count+=len(chunk)
                sub=chunk[chunk.user_session.isin(sessions)].merge(maps,on=['user_id','user_session'],validate='many_to_one')
                if len(sub):matches.append(sub.assign(source_month=month))
        metadata.append(dict(month=month,source_url=url+month+'.csv',sha256=sha(path),rows=count))
        print('boundary scanned',month,count,flush=True)
    match=pd.concat(matches,ignore_index=True) if matches else pd.DataFrame(columns=['session_id_hash','source_month'])
    match.to_csv(HERE/'Cosmetics/cross_month_events.csv.gz',index=False)
    crossing=sorted(set(match.session_id_hash))
    save(HERE/'Cosmetics/cross_month_sessions.json',crossing)
    save(HERE/'Cosmetics/boundary_followup.json',dict(sources=metadata,crossing_sessions=len(crossing),seconds=time.perf_counter()-start,policy='Exclude any original December session key (user_id,user_session) observed in any other released month. Preserve original split and December labels for remaining sessions. No event features or outcome-based selection from added months.',limitation='Session closure inferred from session identifiers plus absence throughout adjacent released months; no explicit session-end events. Tracking gaps remain possible.'))

if __name__=='__main__':main()
