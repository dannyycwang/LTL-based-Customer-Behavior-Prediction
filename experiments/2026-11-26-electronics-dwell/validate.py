from common import *
def main():
    from evaluate import gate
    gate();ds='Electronics';d=load(privdir(ds)/'prepared.pkl');prep=json.loads((outdir(ds)/'preparation.json').read_text())
    assert sha(privdir(ds)/'prepared.pkl')==prep['prepared_sha256']
    for s,x in d.items():
        f=pd.read_csv(privdir(ds)/f'cohort_t20_{s}.csv.gz',dtype={'session_id':str,'user_id':str});np.testing.assert_array_equal(f.session_id,x['session_id']);np.testing.assert_array_equal(f.label,x['label']);assert np.isnan(x['gaps'][:,-1]).all();assert np.isfinite(x['time']).all()
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(d[a]['session_id'])&set(d[b]['session_id'])
    assert len(list(outdir(ds).glob('M[234]_4[234]/freeze.json')))==9
    m=pd.read_csv(HERE/'metrics.csv');p=pd.read_csv(HERE/'paired_differences.csv');assert len(m)==45 and len(p)==48
    assert np.isfinite(m[['value','lower_95','upper_95']]).all().all();assert np.isfinite(p[['delta','lower_95','upper_95']]).all().all();assert (p.valid_replicates==1500).all();assert set(p.unit)=={'user'}
    for c in ['M1','M3','M4','M5']:assert len(json.loads((outdir(ds)/c/'selection.json').read_text())['indices'])<=10
    am=pd.read_csv(HERE/'three_dataset_metrics.csv');ap=pd.read_csv(HERE/'three_dataset_paired_differences.csv');assert len(am)==135 and len(ap)==144
    previous=HERE.parent/'2026-11-23-dwell-time'
    for filename,combined in [('metrics.csv',am),('paired_differences.csv',ap)]:
        before=pd.read_csv(previous/filename);pd.testing.assert_frame_equal(before.reset_index(drop=True),combined[combined.dataset!='Electronics'].reset_index(drop=True),check_dtype=False,atol=1e-14,rtol=1e-14)
    save(HERE/'final_validation.json',dict(passed=True,new_metric_rows=45,new_paired_rows=48,combined_metric_rows=135,combined_paired_rows=144,all_9_neural_fits_frozen=True,previous_results_unchanged=True,test_user_clustered=True))
    done=json.loads((HERE/'completion.json').read_text());done['complete']=True;save(HERE/'completion.json',done);print('VALIDATION PASS',flush=True)
if __name__=='__main__':main()
