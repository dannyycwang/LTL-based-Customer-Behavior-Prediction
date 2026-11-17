from common import *
import re,shutil,zipfile,platform,importlib.metadata

def main():
    results=HERE/'results';source=HERE/'cosmetics_source';assert (source/'comparison/metrics.csv').exists()
    assert len(list((HERE/'neural').glob('*/metrics.json')))==36
    log=(HERE/'relearn_cpu.log').read_text(encoding='utf-8',errors='replace')
    match=re.search(r'Training finished: pack=BP, restart=0, time=([\d.]+)s, wall=([\d.]+)s',log)
    assert match
    timing=json.loads((source/'timing.json').read_text());timing['provisional_basis_sha256']=timing.pop('basis_sha256');timing['final_basis_sha256']=sha(source/'basis.json');train=float(match[2]);timing.update(training_wall_seconds_from_log=train,remaining_setup_interpretation_ranking_serialization_seconds=timing['learning_and_ranking_seconds']-train,training_log_precision_seconds=.01,device='CPU',not_a_controlled_hardware_benchmark=True)
    save_json(source/'timing_breakdown.json',timing)
    pd.concat([pd.read_csv(HERE/'prepared'/f'{d}_timing.csv') for d in ['Coveo','Electronics','Cosmetics']]).to_csv(results/'feature_construction_timing.csv',index=False)
    pd.DataFrame([json.loads(p.read_text()) for p in (HERE/'neural').glob('*/metrics.json')])[['dataset','condition','seed','epochs','training_validation_seconds','inference_precomputed_seconds']].to_csv(results/'neural_execution_timing.csv',index=False)
    srcmetrics=pd.read_csv(source/'comparison/metrics.csv');srcmetrics.to_csv(results/'cosmetics_basis_source.csv',index=False);srcmetrics[['condition','MCC','AP']].to_latex(results/'cosmetics_basis_source.tex',index=False,float_format=lambda v:f'{v:.4f}')
    shutil.copy2(source/'comparison/paired_differences.csv',results/'cosmetics_basis_source_paired.csv')
    # Every inherited implementation is referenced by content hash; original sources remain untouched.
    dependencies=HERE/'implementation_snapshot';dependencies.mkdir(exist_ok=True)
    paths=list(AUDIT.glob('*.py'))+[Path('C:/Users/ycw/Downloads')/name for name in ['product_rawid_vs_ltl_experiment.py','ltl_integrated_pytorch_behavior_dwell_automaton_experiment.py','electronics_ltlf_pipeline.py','train_cosmetics_first_window_clean_names_v6_withBD_len20_fullfusion.py','learn_cosmetics_tltlf_formulas_gpu_from_electronics_style_v3.py','prepare_cosmetics_dec_dataset.py']]
    manifest=[]
    for p in paths:
        shutil.copy2(p,dependencies/p.name);manifest.append(dict(path=str(p),sha256=sha(p),snapshot=str(dependencies/p.name)))
    save_json(dependencies/'manifest.json',manifest)
    import torch,sklearn,scipy,matplotlib
    save_json(results/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,gpu=torch.cuda.get_device_name(),platform=platform.platform()))
    # Enumerate formal outputs independently of archived failed/benchmark attempts.
    inventory=[]
    for base in ['neural','linear','cosmetics_source/comparison']:
        for p in (HERE/base).rglob('metrics.json'):inventory.append(dict(family=base,path=str(p),sha256=sha(p),status='completed'))
    pd.DataFrame(inventory).to_csv(results/'completed_experiments.csv',index=False)
    save_json(results/'completion_status.json',dict(neural_runs=36,template_linear_runs=len(list((HERE/'linear').glob('*/*/metrics.json'))),cosmetics_source_linear_runs=2,formal_formula_relearning='completed 60 epochs on CPU, seed 42',main_basis='frozen basis unchanged',not_run=[dict(name='Electronics common-ranker learned K=10 sensitivity',reason='Only four usable empirically unique candidates in the recovered ten-formula pool; no invented or test-selected replacements'),dict(name='prefix-trained formula sensitivity',reason='Not required for the primary source contrast; no observation-scope/target conflation'),dict(name='new fixed-versus-dynamic Full-raw timing',reason='Audited existing Full-raw cost code already uses dynamic per-batch widths')],limitations=['Exact LaTeX of latest attached PDF not recovered; compiled manuscript uses recovered source and latest-PDF replacement text is supplied.','Cosmetics source contrast excludes 45 fitting and 21 ranking histories with no observable pre-purchase event. CPU and gradient-accumulation implementation differ from historical GPU execution.','Original Electronics split-file exact identity remains unverified; historical train/test users overlap.','Historical formula-learning and ranking end-to-end costs are not fully reconstructed for Coveo/Electronics. No overall training speedup claimed.']))
    # Compact review archive: full predictions/checkpoints stay beside the archive.
    archive=HERE/'review_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in HERE.glob('*.py'):z.write(p,p.relative_to(HERE))
        for name in ['protocol.json','README_先讀.txt']:
            if (HERE/name).exists():z.write(HERE/name,name)
        for folder in ['results','audit']:
            for p in (HERE/folder).rglob('*'):
                if p.is_file() and 'draft' not in p.name and p.name!='review_bundle_manifest.json':z.write(p,p.relative_to(HERE))
        for name in ['basis.json','final_selection.json','timing_breakdown.json','clean_top20.csv','comparison/metrics.csv','comparison/paired_differences.csv']:
            p=source/name;z.write(p,p.relative_to(HERE))
        for p in (HERE/'paper_recovered_revision').rglob('*'):
            if p.is_file() and p.suffix in ['.tex','.bib','.bst','.cls','.pdf']:z.write(p,p.relative_to(HERE))
    save_json(results/'review_bundle_manifest.json',dict(path=str(archive),sha256=sha(archive),excludes='Raw datasets, prepared features, neural checkpoints and detailed predictions remain in sibling prepared/, neural/, linear/, cosmetics_source/ folders.'))
if __name__=='__main__':main()
