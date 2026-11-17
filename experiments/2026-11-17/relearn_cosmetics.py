"""Matched original replayed sample IDs, pre-purchase last-20 formula learning."""
from common import *
import importlib.util,shutil,time
from optimize_learner import patch,validate
from verify_cosmetics_seed42 import notebook_sample
from fix_cosmetics_parser import parse_event_trace

def main():
    out=HERE/'cosmetics_source';out.mkdir(exist_ok=True)
    source=Path('C:/Users/ycw/Downloads/learn_cosmetics_tltlf_formulas_gpu_from_electronics_style_v3.py')
    spec=importlib.util.spec_from_file_location('learner',source);m=importlib.util.module_from_spec(spec);sys.modules['learner']=m;spec.loader.exec_module(m)
    repo=out/'TLTLf'
    if not repo.exists():
        repo.mkdir()
        for name in ['train.py','data.py','model.py','interpret.py']:shutil.copy2(Path('C:/Users/ycw/TLTLf')/name,repo/name)
        m.patch_tltlf_cuda(repo)
        # No original repository is changed. The original adapter exposes its CUDA numpy patch.
        m.patch_interpret_numpy_to_cpu(repo)
        for name in ['train.py','data.py','model.py','interpret.py']:
            p=repo/name
            p.write_text(p.read_text(encoding='utf-8').replace('torch.device("cuda" if torch.cuda.is_available() else "cpu")','torch.device("cpu")'),encoding='utf-8')
        save_json(out/'gradient_equivalence.json',validate(repo))
        patch(repo)
    props=read_basis(PROJECT/'work/cosmetics_BP_basis_clean.json')
    vocab=json.loads((PROJECT/'work/cosmetics_BP_basis_clean.json').read_text())['vocab']
    objs={};audit=[]
    for split,seed in [('train',42),('val',141)]:
        historic=pd.read_csv(PROJECT/'work'/f'cosmetics_seed42_{split}.csv',dtype={'session_id_hash':str,'user_id':str})
        sample=notebook_sample(historic,seed)
        reference=json.loads((PROJECT/'work'/f'cosmetics_bp_{split}_json.json').read_text())
        for label,key in [(1,'traces_pos'),(0,'traces_neg')]:
            replay=[[sorted(s&set(vocab)) for s in parse_event_trace(v)[-20:]] for v in sample.loc[sample.purchase==label,'prop_BDP']]
            assert replay==reference[key], 'Original fitting/ranking ID replay failed'
        before=pd.read_csv(PROJECT/'work/cosmetics_pre_purchase_seed42'/f'{split}.csv',dtype={'session_id_hash':str}).set_index('session_id_hash')
        keep=sample.session_id_hash.isin(before.index);removed=sample.loc[~keep,['session_id_hash','purchase']]
        removed.to_csv(out/f'{split}_zero_observation_exclusions.csv',index=False)
        sample=sample.loc[keep].copy();sample[['session_id_hash','user_id','purchase']].to_csv(out/f'{split}_ids.csv',index=False)
        traces=[parse_event_trace(before.loc[sid,'prop_BP'])[-20:] for sid in sample.session_id_hash]
        obj=dict(vocab=vocab,traces_pos=[[sorted(s&set(vocab)) for s in tr] for tr,y in zip(traces,sample.purchase) if y==1],traces_neg=[[sorted(s&set(vocab)) for s in tr] for tr,y in zip(traces,sample.purchase) if y==0])
        # Original TLTLf JSON metadata keys are preserved.
        for k,v in reference.items():
            if k not in obj:obj[k]=v
        objs[split]=obj;audit.append(dict(split=split,n=len(sample),excluded_empty=len(removed),positive=int(sample.purchase.sum()),ids_sha256=sha(out/f'{split}_ids.csv')))
    save_json(out/'protocol.json',dict(source_sha256=sha(source),samples=audit,epochs=60,netlen=10,lr=.01,batch_size=100,restarts=1,max_candidates=300,seed=42,scope='pre-first-purchase, then last 20',selection_target='eventual session purchase',original_training_seconds=1058.02,original_timing_source='electronics_nb_0_cells.txt lines 634-640',limitation='Empty pre-purchase histories excluded, no replacement; historical training RNG/kernel versions may differ',implementation='Equivalent additive gradient terms combined into one backward; verified on CPU; original GPU attempts archived, not used',device='CPU selected by train-only throughput benchmark; original historical learner used GPU'))
    start=time.perf_counter()
    if (out/'basis.json').exists():return
    result=m.learn_one_pack(pack='BP',vocab=vocab,repo_dir=repo,out_dir=out,train_obj=objs['train'],val_obj=objs['val'],netlen=10,epochs=60,lr=.01,batch_size=100,restarts=1,max_candidates=300,select_k=20,seed=42,reuse_models=False)
    rows=read_basis(result.selected_json)
    def leaves(t):return {t} if isinstance(t,str) else set().union(*(leaves(c) for c in t[1:]))
    rows=[r for r in rows if leaves(tree(r))&set(vocab)]
    assert len(rows)>=10
    save_json(out/'basis.json',rows[:10]);save_json(out/'timing.json',dict(learning_and_ranking_seconds=time.perf_counter()-start,source='new local run; includes interpretation/ranking',basis_sha256=sha(out/'basis.json')))

if __name__=='__main__':main()
