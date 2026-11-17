from common import *
import time
sys.path.insert(0,str(HERE/'cosmetics_source/TLTLf'))
import torch,train,data,model
for m in [train,data,model]:m.device=torch.device('cpu')
out=HERE/'smoke/learner_cpu_fused';out.mkdir(parents=True,exist_ok=True)
obj=json.loads((HERE/'cosmetics_source/BP/data/train.json').read_text())
obj['traces_pos']=obj['traces_pos'][:50];obj['traces_neg']=obj['traces_neg'][:50]
save_json(out/'train.json',obj)
args=train.get_train_argument(['--train_file',str(out/'train.json'),'--save_model',str(out/'model.pt'),'--log_file',str(out/'logs'),'--netlen','10','--epoch','2','--lr','.01','--batch_size','100','--test_correctness','0'])
t=time.perf_counter();elapsed=train.train(args);save_json(out/'timing.json',dict(two_updates_seconds=time.perf_counter()-t,training_seconds=elapsed,status='benchmark only, not formal results'))
