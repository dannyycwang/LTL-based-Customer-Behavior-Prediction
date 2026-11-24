"""Bounded matrix, failure-preserving runner. Never starts an extra seed or condition."""
import sys,subprocess,json,time,ctypes,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
def run(args):
    print('RUN',args,flush=True)
    with (HERE/(args[0].replace('.py','')+'_'+('_'.join(args[1:]) or 'all')+'.log')).open('a',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
def wait_existing():
    import psutil
    processes=json.loads((HERE/'private/processes.json').read_text())
    while True:
        pending=[]
        for item in processes:
            marker=HERE/item['marker']
            if marker.exists():continue
            pid=item['pid']
            if not psutil.pid_exists(pid) or abs(psutil.Process(pid).create_time()-item['created'])>1:
                raise RuntimeError(f'Existing process {pid} stopped without {marker}; inspect logs, do not replace seed')
            pending.append(item)
        if not pending:return
        time.sleep(15)
def main():
    # Keep system awake only while this finite experiment is active. No settings changed.
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if '--attached' in sys.argv:
            wait_existing()
        else:
            run(['inventory.py']);run(['prepare.py']);run(['check_features.py'])
            for ds in ['Coveo','Cosmetics']:run(['mine.py',ds]);run(['linear.py',ds])
        for ds in ['Coveo','Cosmetics']:run(['neural.py',ds])
        run(['evaluate.py']);run(['report.py']);run(['validate.py'])
        print('COMPLETE fixed matrix; no additional research scheduled',flush=True)
    except Exception as e:
        (HERE/'failure.json').write_text(json.dumps(dict(error=repr(e),time=time.time()),indent=2),encoding='utf-8')
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
