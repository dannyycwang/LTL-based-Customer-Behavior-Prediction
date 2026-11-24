"""Locate historical artifacts; export aggregate-only evidence and file hashes."""
from common import *
from zipfile import ZipFile
import platform,importlib.metadata

def main():
    home=Path('C:/Users/ycw');download=home/'Downloads';records=[]
    # Search the actual local locations, not only the current README.
    roots=[HERE.parents[1],old.SOURCE,download,home/'behavior_ltl_outputs',home/'Dataset/Clickstream',home/'runs',home/'output',home/'TLTLf',home/'Formulasetforlengthlast30']
    found=[]
    for root in roots:
        if not root.exists():continue
        for parent,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in ['.git','python_packages','__pycache__','node_modules','private','vendor_liblinear']]
            for name in files:
                if any(s in name.lower() for s in ['dwell','cosmetics_first_window_results','colab_','electronics_nb','experiment results_dataset1']):
                    p=Path(parent)/name
                    found.append(dict(path=str(p),bytes=p.stat().st_size))
    save(PRIVATE/'historical_search_manifest.json',dict(roots=[str(p) for p in roots],matches=found))
    files=[download/'experiment results_dataset1.zip',download/'cosmetics_first_window_results_clean_names_v6_withBD_len20_fullfusion_dataset3.csv',download/'build_d_raw_random4_duckdb_streaming_lowmem_v2.py',download/'electronics_ltlf_pipeline.py',download/'prepare_cosmetics_dec_dataset.py',download/'Dwell_top20_formulas.json',download/'Z_BD_top20_formulas.json']
    files+=list(old.WORK.glob('colab_*.ipynb'))+list(old.WORK.glob('electronics_nb_*.ipynb'))+[HERE.parents[1]/'PCS955experiment.ipynb']
    save(HERE/'historical_sources.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files if p.exists()])
    archive=download/'experiment results_dataset1.zip'
    with ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('.csv') and '/all_runs' in name and ('with_dwell' in name or 'behavior_dwell' in name):
                df=pd.read_csv(z.open(name))
                keep=[c for c in df if c.lower() in ['seed','model','mode','raw_length','raw_len','length','top_k','k','f1','mcc','auc','auprc','brier','run_name','condition']]
                for row in df[keep].to_dict('records'):records.append(dict(dataset='Coveo',source_member=name,**row))
    cos=pd.read_csv(download/'cosmetics_first_window_results_clean_names_v6_withBD_len20_fullfusion_dataset3.csv')
    cos=cos[cos.raw_pack.fillna('').str.contains('D')|cos.z_pack.fillna('').str.contains('D')]
    for row in cos[[c for c in ['seed','run','mode','raw_pack','z_pack','raw_len','z_top_k','F1','MCC','AUC','AUPRC','Brier'] if c in cos]].to_dict('records'):records.append(dict(dataset='Cosmetics',source_member='downloaded historical CSV',**row))
    pd.DataFrame(records).to_csv(HERE/'historical_aggregate_results.csv',index=False)
    env=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib','tabulate','psutil','threadpoolctl']},prior_solver='vendor_liblinear/liblinear 2.50.0',search_roots=[str(p) for p in roots],search_matches=len(found))
    import torch
    env.update(cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    save(HERE/'environment.json',env)
    print('Historical search matches',len(found),'aggregate rows',len(records),flush=True)
if __name__=='__main__':main()
