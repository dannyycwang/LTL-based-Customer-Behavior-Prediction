import sys,subprocess,time,json,ctypes,os
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(args):
    print('RUN',args,flush=True)
    with (HERE/(args[0].replace('.py','')+'_'+('_'.join(args[1:]) or 'all')+'.log')).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-X','utf8','-u',*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        run(['prepare.py']);run(['check_features.py']);run(['mine.py','actrel']);run(['mine.py','dwell']);run(['linear.py']);run(['neural.py','Electronics']);run(['evaluate.py']);run(['report.py']);run(['validate.py']);print('COMPLETE fixed Electronics extension',flush=True)
    except Exception as e:
        (HERE/'failure.json').write_text(json.dumps(dict(error=repr(e),time=time.time()),indent=2));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
