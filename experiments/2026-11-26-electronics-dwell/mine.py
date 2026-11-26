from common import *
import random

def main(vocabulary):
    ds='Electronics';dest=outdir(ds);private=privdir(ds);recipe=CFG['mining'];name=vocabulary+'_candidates.json'
    if (dest/name).exists():return
    marker=dest/(vocabulary+'_mining_started.json')
    if marker.exists():raise RuntimeError('Existing mining attempt; inspect before any restart')
    sys.path.insert(0,str(PREV/'vendor'))
    import torch
    from train import train,get_train_argument
    from interpret import matrix2ltl
    torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    file=private/(vocabulary+'_train.json');model=private/(vocabulary+'_model.pt')
    save(marker,dict(seed=42,recipe=recipe,input_sha256=sha(file),source_hashes={n:sha(PREV/'vendor'/n) for n in ['data.py','model.py','interpret.py','train.py']}))
    start=time.perf_counter();args=get_train_argument(['--train_file',str(file),'--save_model',str(model),'--log_file',str(private/(vocabulary+'_log')),'--netlen','10','--epoch','50','--lr','0.01','--batch_size','100'])
    train(args);trees=matrix2ltl(str(model),str(file),250)
    save(dest/name,[dict(formula_tree=t,formula_str=repr(t),candidate=i,source=vocabulary+'_matched_training') for i,t in enumerate(trees)])
    save(dest/(vocabulary+'_mining_complete.json'),dict(seconds=time.perf_counter()-start,candidates=len(trees),model_sha256=sha(model),candidates_sha256=sha(dest/name)))
    timing(ds,'mining',time.perf_counter()-start,vocabulary=vocabulary)
if __name__=='__main__':main(sys.argv[1])
