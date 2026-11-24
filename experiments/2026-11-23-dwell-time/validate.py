from common import *
from evaluate import gate
def main():
    gate()
    m=pd.read_csv(HERE/'metrics.csv');p=pd.read_csv(HERE/'paired_differences.csv')
    assert len(m)==90 and len(p)==96,(len(m),len(p))
    assert np.isfinite(m[['value','lower_95','upper_95']]).all().all()
    assert np.isfinite(p[['delta','lower_95','upper_95']]).all().all()
    assert set(m.dataset)==set(CFG['datasets'])
    for ds in CFG['datasets']:
        previous_metrics=pd.read_csv(PREV/ds/'metrics.csv')
        for current,previous in [('M0','A'),('M1','C')]:
            for metric in ['AP','MCC','Brier']:
                now=m[(m.dataset==ds)&(m.model=='LR')&(m.condition==current)&(m.metric==metric)].value.iloc[0]
                before=previous_metrics[(previous_metrics.landmark==20)&(previous_metrics.condition==previous)&(previous_metrics.metric==metric)].value.iloc[0]
                np.testing.assert_allclose(now,before,atol=1e-12,rtol=0)
        d=load(privdir(ds)/'prepared.pkl');prep=json.loads((outdir(ds)/'preparation.json').read_text())
        assert sha(privdir(ds)/'prepared.pkl')==prep['prepared_sha256']
        for s,x in d.items():
            c=pd.read_csv(PREV/ds/f'cohort_t20_{s}.csv.gz',dtype={'session_id':str,'user_id':str})
            np.testing.assert_array_equal(c.session_id,x['session_id']);np.testing.assert_array_equal(c.label,x['label'])
            np.testing.assert_array_equal(c.user_id.astype(str),x['user_id'].astype(str))
            assert np.isnan(x['gaps'][:,-1]).all() and np.isfinite(x['time']).all()
        if ds=='Cosmetics':
            for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(d[a]['user_id'])&set(d[b]['user_id'])
        for cond in ['M3','M4','M5']:
            sel=json.loads((outdir(ds)/cond/'selection.json').read_text());assert len(sel['indices'])<=10
        assert len(list(outdir(ds).glob('M[234]_4[234]/freeze.json')))==9
    save(HERE/'final_validation.json',dict(passed=True,metrics_rows=len(m),paired_rows=len(p),all_18_neural_fits_frozen=True,same_cohorts=True,reused_M0_M1_metrics_match_previous_to_1e_12=True,private_artifacts_excluded_by_gitignore=True))
    done=json.loads((HERE/'completion.json').read_text());done.update(complete=True,awaiting_final_validation=False);save(HERE/'completion.json',done)
    readme=(HERE/'README.md').read_text(encoding='utf-8').replace('Execution in progress. Do not treat this directory as completed results until `completion.json` and `final_validation.json` both exist and report success.','Fixed matrix completed; `completion.json` and `final_validation.json` report success.')
    (HERE/'README.md').write_text(readme,encoding='utf-8')
    print('Final validation passed',flush=True)
if __name__=='__main__':main()
