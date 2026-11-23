"""Artifact consistency checks after the sole completed pilot."""
from pilot import *
import sklearn,scipy,torch,platform

def main():
    metrics=pd.read_csv(HERE/'metrics.csv');diff=pd.read_csv(HERE/'paired_differences.csv')
    price=json.loads((HERE/'price_cleaning_audit.json').read_text())
    assert price['all_eligible_prefixes_and_future_first_purchase_labels_unchanged_by_cleaning']
    closure=json.loads((HERE/'coveo_search_closure_audit.json').read_text())
    assert closure['all_retained_sessions_pass_combined_activity_boundary_check']
    assert len(metrics)==72 and len(diff)==54
    assert set(metrics.condition)==set('ABCD') and set(metrics.metric)=={'AP','MCC','Brier'}
    frozen=[];predicted=[];checks=[]
    for ds in CFG['datasets']:
        pool=HERE/ds/'mining';m=json.loads((pool/'manifest.json').read_text())
        assert m['status']=='complete' and m['seed']==42 and m['candidates_sha256']==sha(pool/'candidates.json')
        assert sha(pool/'train.json')==m['input_sha256']
        for t in CFG['landmarks']:
            dest=HERE/ds/f't{t}';freeze_=json.loads((dest/'freeze.json').read_text())
            assert freeze_['model_sha256']==sha(dest/'frozen.pkl')
            assert freeze_['config_sha256']==sha(HERE/'config.json')
            frozen.append((dest/'freeze.json').stat().st_mtime)
            predicted.append((dest/'predictions.csv.gz').stat().st_mtime)
            p=pd.read_csv(dest/'predictions.csv.gz',dtype={'session_id':str,'user_id':str})
            cohort=pd.read_csv(HERE/ds/f'cohort_t{t}_test.csv.gz',dtype={'session_id':str,'user_id':str})
            assert p[['session_id','user_id','label']].equals(cohort[['session_id','user_id','label']])
            assert not p.session_id.duplicated().any() and p[list('ABCD')].notna().all().all()
            if t==10:np.testing.assert_array_equal(p.B,p.C)
            for c in ('B','C','D'):
                sel=json.loads((dest/c/'selection.json').read_text());assert len(sel['indices'])<=10
                assert len(set(sel['indices']))==len(sel['indices'])
                steps=dest/c/'selection_steps.csv'
                if steps.exists():
                    trace=pd.read_csv(steps);accepted=trace[trace.accepted]
                    assert len(accepted)==len(sel['indices'])
                    assert (accepted.gain>CFG['minimum_logloss_gain']).all()
                    assert (trace[~trace.accepted].gain<=CFG['minimum_logloss_gain']).all()
                assert sel['frozen']
            checks.append(dict(dataset=ds,landmark=t,rows=len(p),positives=int(p.label.sum()),shared_cohort=True))
    assert max(frozen)<=min(predicted),'Test predictions preceded complete freeze'
    save(HERE/'final_validation.json',dict(status='passed',checks=checks,all_model_formula_freezes_before_first_test_prediction=True,
        test_used_previously_in_research=True,bootstrap_models_and_formulas_fixed=True,report_scope='logistic regression only; exploratory'))
    save(HERE/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,torch=torch.__version__,classifier_backend=CFG['classifier_backend'],native_tolerance=CFG['native_tolerance']))
    files={}
    for path in HERE.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ['private_boundary_sources','__pycache__','mining_failed_console_encoding','vendor_liblinear']):continue
        if path.suffix in ('.py','.json','.csv','.md','.png','.svg','.ps1') and path.name not in ('checksums.json','train.json','cross_month_sessions.json'):
            files[str(path.relative_to(HERE)).replace('\\','/')]=sha(path)
    save(HERE/'checksums.json',files)
    print('Final artifact validation passed',flush=True)

if __name__=='__main__':main()
