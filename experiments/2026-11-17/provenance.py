from common import *
import shutil,platform
from verify_cosmetics_seed42 import notebook_sample
from fix_cosmetics_parser import parse_event_trace

def main():
    out=HERE/'audit/provenance';out.mkdir(parents=True,exist_ok=True);sources=[]
    paths=[AUDIT/'coveo_formula_sample_provenance.json',AUDIT/'cosmetics_seed42_audit/manifest.json',
           AUDIT/'cosmetics_pre_purchase_seed42_comparison/protocol.md',AUDIT/'electronics_rebuild/manifest.json',
           AUDIT/'submission_controls_seed42/inference_runtime_protocol.json',
           AUDIT/'runs/protected_first_v1_seed42_short_raw/protected_formula_pool.csv',
           PROJECT/'work/electronics_rebuild/formula_learning_noleak_4000_last10_seed42/BP/formula_pool_ids.csv']
    for i,p in enumerate(paths):
        target=out/f'{i:02}_{p.name}';shutil.copy2(p,target);sources.append(dict(path=str(p),copied_to=str(target),sha256=sha(p)))
    replay=[]
    for split,seed in [('train',42),('val',141)]:
        path=PROJECT/'work'/f'cosmetics_seed42_{split}.csv';frame=pd.read_csv(path,dtype={'session_id_hash':str,'user_id':str})
        sample=notebook_sample(frame,seed);vocab=json.loads((PROJECT/'work/cosmetics_BP_basis_clean.json').read_text())['vocab']
        original=json.loads((PROJECT/'work'/f'cosmetics_bp_{split}_json.json').read_text())
        for label,key in [(1,'traces_pos'),(0,'traces_neg')]:
            assert [[sorted(s&set(vocab)) for s in parse_event_trace(v)[-20:]] for v in sample.loc[sample.purchase==label,'prop_BDP']]==original[key]
        target=out/f'Cosmetics_{split}_actual_replayed_ids.csv';sample[['session_id_hash','user_id','purchase']].to_csv(target,index=False)
        replay.append(dict(split=split,n=len(sample),ids=str(target),ids_sha256=sha(target),trace_order_exact=True))
    specs={
      'Coveo':dict(fitting_ids=str(paths[5]),ranking_ids=str(paths[5]),scope='last 30 from historical pre-purchase traces',selection_target='session purchase',learner_seed=42,source_status='exact 20000 input-trace replay verified in prior audit; protected downstream split reconstructed'),
      'Electronics':dict(fitting_ids=str(paths[6]),ranking_ids=str(paths[6]),scope='last 10 of pre-purchase traces',selection_target='session purchase',learner_seed=42,source_status='8000 train formula-pool sessions reconstructed, ten ranking support pairs replayed; original split CSV exact identity unverified'),
      'Cosmetics':dict(samples=replay,scope='last 20 after removing purchases; historical traces can extend beyond first purchase',selection_target='session purchase',learner_seed=42,epochs=60,netlen=10,ranking='balanced accuracy, absolute support gap, positive support, candidate order; empirical validation-vector deduplication',post_purchase_fitting_histories=526,post_purchase_ranking_histories=259,test_in_formula_pools=False,source_status='trace-matching replayed IDs used; saved sample CSV IDs differ and are not used as actual learner IDs')}
    specs['Cosmetics']['ranking']='Candidate learner ranking followed by actual notebook clean_pack: prefer all_candidates; remove length-only; sort balanced_accuracy, signed support_gap, pos_support descending; drop identical formula strings; first ten of clean top20. Empirical deduplication is not guaranteed by this cleanup.'
    save_json(out/'formula_provenance.json',specs);save_json(out/'source_manifest.json',sources)
    save_json(out/'cost_audit.json',dict(
        full_raw_padding={'Coveo':'trim_full_batch: trim trailing shared padding, no length bucketing','Electronics':'per-batch dynamic left-padding trim with original position-ID offset','Cosmetics':'per-batch dynamic right-padding trim, no length bucketing'},
        evidence=['run_coveo.py:trim_full_batch','run_act_rel_seed42.py:to_device,model_for','benchmark_training_compute.py:DataLoader collate_fn','benchmark_submission_runtime.py:timing scope'],
        fixed_padding_confound='published audited cost measurements already use dynamic batch widths; no new fixed-vs-dynamic efficiency claim',
        historical_inference_includes=['loaded-field parsing','raw packing','summary construction/checking','model and transfer'],
        historical_inference_excludes=['disk I/O','original event abstraction','formula learning','model loading'],
        historical_formula_learning={'Cosmetics_BP_seconds':1058.02,'source':'electronics_nb_0_cells.txt lines 634-640','Coveo':'not newly verified as full learning+ranking total','Electronics':'not newly verified as full learning+ranking total'},
        new_timings='prepared/*_timing.csv separate parsing/cohort, Counts, all-candidate Transitions, all-candidate formula checks and templates. These are not selected-K=10 latency benchmarks and exclude original abstraction.',
        neural_and_learner_concurrency='Some new neural runs overlap the new Cosmetics learner; wall times are execution logs, not controlled efficiency comparisons. No overall training speedup claimed.',
        environment=dict(python=sys.version,platform=platform.platform())))
    # Document observed equivalence to occurrence features without claiming logical equivalence.
    rows=[]
    for ds in specs:
        data=load_data(ds);meta=json.loads((HERE/'prepared'/f'{ds}_manifest.json').read_text())
        for j in range(10):
            for k,atom in enumerate(meta['props']):
                if all(np.array_equal(d['ltlf'][:,j],d['counts'][:,k]>0) for s,d in data.items() if s!='test'):
                    rows.append(dict(dataset=ds,formula=j+1,equal_to_occurrence_of=atom,scope='first 10, train and validation only',logical_claim=False))
    pd.DataFrame(rows).to_csv(HERE/'audit/empirical_occurrence_matches.csv',index=False)
if __name__=='__main__':main()
