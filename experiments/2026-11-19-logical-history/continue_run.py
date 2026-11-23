"""Resume this one pilot as existing mining jobs finish; never starts another mining run."""
import json,time,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
jobs={};handles=[];done=set();start=time.perf_counter()
while len(done)<2:
    for ds in ('Coveo','Cosmetics'):
        if ds in done:continue
        if ds in jobs:
            status=jobs[ds].poll()
            if status is not None:
                if status:raise RuntimeError(f'{ds} selection failed with {status}; inspect fit log')
                done.add(ds);print(ds,'selection frozen',flush=True)
        else:
            # Memory-bounded: one dataset selector, four concurrent candidate fits.
            if jobs and any(job.poll() is None for job in jobs.values()):continue
            path=HERE/ds/'mining/manifest.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='complete':
                handle=(HERE/f'{ds.lower()}_fit.log').open('a');handles.append(handle)
                jobs[ds]=subprocess.Popen([sys.executable,'-u',str(HERE/'fit.py'),'fit',ds],stdout=handle,stderr=subprocess.STDOUT)
                print(ds,'selection started',flush=True)
    time.sleep(5)
for ds in ('Coveo','Cosmetics'):
    with (HERE/f'{ds.lower()}_evaluation.log').open('w') as handle:
        subprocess.run([sys.executable,'-u',str(HERE/'fit.py'),'evaluate',ds],stdout=handle,stderr=subprocess.STDOUT,check=True)
    print(ds,'evaluation completed',flush=True)
subprocess.run([sys.executable,'-u',str(HERE/'report.py')],check=True)
for handle in handles:handle.close()
print('Pilot numerical pipeline completed',time.perf_counter()-start,'seconds',flush=True)
