"""Read-only source audit. Writes derived outputs beside this script."""
from pathlib import Path
import hashlib, json, platform
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
HOME = Path('C:/Users/ycw')
DOWNLOADS = HOME / 'Downloads'
RESULTS = HOME / 'OneDrive - Høgskulen på Vestlandet/experiment results'
LAB = HOME / 'OneDrive - Høgskulen på Vestlandet/Desktop/LTL experiment'
METRICS = ['f1', 'mcc', 'auprc', 'brier']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    entries = []
    def add(dataset, scheme, condition, path, select, paper, raw_branch):
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        sub = select(df).copy()
        if sorted(sub.seed.tolist()) != [42,43,44,45]:
            raise ValueError(f'Unexpected seeds/duplicates: {dataset} {scheme} {condition}: {sub.seed.tolist()}')
        for _, r in sub.iterrows():
            entries.append(dict(dataset=dataset, scheme=scheme, condition=condition,
                seed=int(r.seed), raw_branch=raw_branch, source=str(path), source_sha256=sha256(path),
                source_row_1based=int(r.name)+2, **{m:float(r[m]) for m in METRICS},
                **{f'paper_{m}':v for m,v in zip(METRICS,paper)}))
    for scheme, folder, file, rawpack, fusionpack, papers in [
        ('Act+D','RawBehavior_Dwell','all_runs_fixed_behavior_dwell_fullMC_top10.csv','raw_dwell','raw_dwell_plus_ltl',
         [[.907,.884,.928,.038],[.872,.841,.927,.048],[.914,.893,.941,.035]]),
        ('Act+Rel','Include_product','all_runs_pc_no_dwell_fixedF42_fullcheck_top10.csv','apc_raw','action_ltl',
         [[.897,.871,.904,.045],[.847,.808,.901,.057],[.912,.890,.928,.038]])]:
        p = RESULTS/folder/file
        for condition,length,pack,paper in zip(['Full raw','Short raw','Short+Full-LTLf'],['full','10','10'],[rawpack,rawpack,fusionpack],papers):
            # feature_pack is the run tag in the Coveo output; experiment_type identifies input.
            add('Coveo',scheme,condition,p,lambda d,l=length,k=pack:d[(d.experiment_type==k)&(d.length_label.astype(str)==l)],paper,
                'action only' if pack=='action_ltl' else ('action+dwell' if scheme=='Act+D' else 'action+product ID+category ID'))
    p=LAB/'dataset2/combined_all_runs (2).csv'
    for scheme,rawpack,ltlpack,papers in [
        ('Act+D','bd_raw','bd_ltl',[[.745,.694,.720,.091],[.734,.680,.715,.096],[.738,.688,.713,.094]]),
        ('Act+Rel','apc_raw','apc_ltl',[[.754,.701,.722,.091],[.741,.686,.711,.097],[.754,.701,.712,.093]])]:
        for condition,length,pack,paper in zip(['Full raw','Short raw','Short+Full-LTLf'],['full','10','10'],[rawpack,rawpack,ltlpack],papers):
            add('Electronics',scheme,condition,p,lambda d,l=length,k=pack,s=scheme:d[(d.feature_pack==k)&(d.length_label.astype(str)==l)&(~d.run_name.str.contains('Z_BDP') if s=='Act+D' else pd.Series(True,index=d.index))],paper,
                'action+dwell' if scheme=='Act+D' else 'action+product ID+category ID')
    p=DOWNLOADS/'cosmetics_first_window_results_clean_names_v6_withBD_len20_fullfusion_dataset3.csv'
    df=pd.read_csv(p); print('Cosmetics runs:',sorted(df.run.unique()))
    for scheme,pack,papers in [
        ('Act+D','BD',[[.707,.648,.727,.106],[.701,.642,.679,.114],[.708,.649,.707,.108]]),
        ('Act+Rel','BP',[[.654,.570,.630,.130],[.657,.571,.609,.128],[.693,.624,.655,.115]])]:
        for condition,run,paper in zip(['Full raw','Short raw','Short+Full-LTLf'],[f'{pack}full_raw',f'{pack}10_raw',f'{pack}10_Z_{pack}_top10'],papers):
            add('Cosmetics',scheme,condition,p,lambda d,k=run:d[d.run==k],paper,
                'joint event token: action+dwell bin' if pack=='BD' else 'joint event token: action+product ID+category ID')
    runs=pd.DataFrame(entries); runs.to_csv(OUT/'table4_source_runs.csv',index=False)
    summary=[]
    for keys,g in runs.groupby(['dataset','scheme','condition'],sort=False):
        row=dict(zip(['dataset','scheme','condition'],keys)); row.update(n_seeds=len(g),source=g.source.iloc[0],raw_branch=g.raw_branch.iloc[0])
        for m in METRICS:
            row[m+'_mean']=g[m].mean();row[m+'_sd']=g[m].std(ddof=1)
            row['paper_'+m]=g['paper_'+m].iloc[0]
            row[m+'_mean_matches_3dp']=round(row[m+'_mean'],3)==row['paper_'+m]
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'table4_correspondence.csv',index=False)
    files=list(dict.fromkeys(runs.source.tolist()))+[str(DOWNLOADS/'Z_APC_top20_formulas.json')]
    manifest=[dict(path=p,bytes=Path(p).stat().st_size,sha256=sha256(p)) for p in files]
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(summary)[['dataset','scheme','condition','f1_mean','mcc_mean','auprc_mean']].to_string(index=False))

if __name__=='__main__': main()
