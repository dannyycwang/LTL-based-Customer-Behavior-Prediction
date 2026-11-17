"""Recompute paper Tables 5-7 from saved per-seed CSVs. No training."""
from audit_sources import *

def main():
    runs=pd.read_csv(OUT/'table4_source_runs.csv')
    paper_sds=[
        [.004,.005,.007,.002],[.003,.004,.002,.001],[.001,.002,.004,.001],
        [.002,.002,.001,.004],[.003,.004,.005,.003],[.002,.003,.003,.002],
        [.007,.005,.009,.004],[.005,.005,.008,.004],[.006,.007,.007,.003],
        [.006,.007,.006,.004],[.007,.008,.004,.001],[.005,.005,.009,.004],
        [.005,.006,.004,.002],[.007,.007,.010,.008],[.006,.006,.008,.004],
        [.008,.010,.012,.006],[.008,.011,.005,.004],[.005,.005,.006,.005]]
    corr=pd.read_csv(OUT/'table4_correspondence.csv')
    for i,sds in enumerate(paper_sds):
        for m,sd in zip(METRICS,sds):
            corr.loc[i,'paper_'+m+'_sd']=sd
            corr.loc[i,m+'_sd_matches_3dp']=round(corr.loc[i,m+'_sd'],3)==sd
    corr.to_csv(OUT/'table4_correspondence.csv',index=False)
    rows=[]
    for dataset,scheme,paper in [
        ('Coveo','Act+Rel',[.065,.082,.027,-.019]),('Coveo','Act+D+Rel',[.046,.058,.021,-.015]),
        ('Electronics','Act+Rel',[.013,.015,.001,-.004]),('Electronics','Act+D+Rel',[.005,.008,.002,-.001]),
        ('Cosmetics','Act+Rel',[.036,.054,.045,-.013]),('Cosmetics','Act+D+Rel',[.043,.061,.051,-.017])]:
        if scheme=='Act+Rel':
            base=runs[(runs.dataset==dataset)&(runs.scheme==scheme)]
            raw=base[base.condition=='Short raw'].set_index('seed');fusion=base[base.condition=='Short+Full-LTLf'].set_index('seed')
        else:
            if dataset=='Coveo':
                d=pd.read_csv(RESULTS/'Include_product/all_runs_pc_with_dwell_fixedF42_fullcheck_top10.csv')
                raw=d[(d.experiment_type=='adpc_raw')&(d.length_label.astype(str)=='10')].set_index('seed')
                fusion=d[(d.experiment_type=='action_dwell_ltl')&(d.length_label.astype(str)=='10')].set_index('seed')
            elif dataset=='Electronics':
                d=pd.read_csv(LAB/'dataset2/combined_all_runs (2).csv')
                raw=d[d.run_name=='ADPC10_raw'].set_index('seed');fusion=d[d.run_name=='ADPC10_Z_BDP_K10'].set_index('seed')
            else:
                d=pd.read_csv(DOWNLOADS/'cosmetics_first_window_results_clean_names_v6_withBD_len20_fullfusion_dataset3.csv');d.columns=d.columns.str.lower()
                raw=d[d.run=='BDP10_raw'].set_index('seed');fusion=d[d.run=='BDP10_Z_BDP_top10'].set_index('seed')
        if len(raw)!=4 or len(fusion)!=4:raise ValueError((dataset,scheme,len(raw),len(fusion)))
        delta=fusion[METRICS]-raw[METRICS]
        row=dict(dataset=dataset,scheme=scheme,status='historical arithmetic only; Coveo raw branches differ')
        for m,p in zip(METRICS,paper):row.update({m+'_delta_mean':delta[m].mean(),'paper_'+m+'_delta':p,m+'_matches_3dp':round(delta[m].mean(),3)==p})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'table5_correspondence.csv',index=False)
    rows=[]
    for dataset,paper in [('Coveo',[.888,.860,.832]),('Electronics',[.737,.686,.678]),('Cosmetics',[.703,.645,.679])]:
        if dataset=='Coveo':d=pd.read_csv(RESULTS/'RawBehavior_Dwell/all_runs_fixed_behavior_dwell_fullMC_top10.csv');s=d[d.experiment_type=='verification_only']
        elif dataset=='Electronics':d=pd.read_csv(LAB/'dataset2/combined_all_runs (2).csv');s=d[d.run_name=='Z_BD_K10_only']
        else:d=pd.read_csv(DOWNLOADS/'cosmetics_first_window_results_clean_names_v6_withBD_len20_fullfusion_dataset3.csv');d.columns=d.columns.str.lower();s=d[d.run=='Z_BD_top10_only']
        if len(s)!=4:raise ValueError(dataset)
        row=dict(dataset=dataset,n_seeds=4)
        for m,p in zip(METRICS[:3],paper):row.update({m+'_mean':s[m].mean(),m+'_sd':s[m].std(ddof=1),'paper_'+m:p,m+'_matches_3dp':round(s[m].mean(),3)==p})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'table6_correspondence.csv',index=False)
    rows=[]
    for scheme,folder,file,rawtype,fustype,paper in [
        ('Act','RawBehavior','all_runs_fixed_formula_fullMC_top10.csv','raw','raw_plus_ltl',[105.5,44.4,84.3]),
        ('Act+D','RawBehavior_Dwell','all_runs_fixed_behavior_dwell_fullMC_top10.csv','raw_dwell','raw_dwell_plus_ltl',[119.1,41.9,72.4]),
        ('Act+Rel','Include_product','all_runs_pc_no_dwell_fixedF42_fullcheck_top10.csv','apc_raw','action_ltl',[159.5,68.6,143.2]),
        ('Act+D+Rel','Include_product','all_runs_pc_with_dwell_fixedF42_fullcheck_top10.csv','adpc_raw','action_dwell_ltl',[167.1,52.9,136.3])]:
        d=pd.read_csv(RESULTS/folder/file);r=d[(d.experiment_type==rawtype)&(d.length_label.astype(str)=='full')];f=d[(d.experiment_type==fustype)&(d.length_label.astype(str)=='10')]
        if len(r)!=4 or len(f)!=4:raise ValueError((scheme,len(r),len(f),d.experiment_type.unique()))
        vals=[r.total_pipeline_cost_sec.mean(),f.model_checking_time_sec.mean(),f.total_pipeline_cost_sec.mean()]
        rows.append(dict(scheme=scheme,raw_total_sec=vals[0],fusion_mc_sec=vals[1],fusion_total_sec=vals[2],paper_raw_total=paper[0],paper_fusion_mc=paper[1],paper_fusion_total=paper[2],matches_1dp=all(round(v,1)==p for v,p in zip(vals,paper)),formula_learning_included=False,status='historical timing; GPU and total-cost scope conflict with manuscript wording'))
    pd.DataFrame(rows).to_csv(OUT/'table7_correspondence.csv',index=False)
    print('Table4 all mean/SD cells match:',corr[[c for c in corr if c.endswith('matches_3dp')]].all().all())
    for t in [5,6,7]:print(t,pd.read_csv(OUT/f'table{t}_correspondence.csv').to_string(index=False))

if __name__=='__main__':main()

