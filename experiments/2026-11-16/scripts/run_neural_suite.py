"""Run the documented 15 neural conditions, optionally the 12 optimization repeats."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path)
parser.add_argument('--initializations',action='store_true')
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--max-minutes-per-run',type=float,default=120)
args=parser.parse_args()
env=dict(os.environ)
if args.config:
    env['LTLF_PATHS']=str(args.config.resolve())
    os.environ['LTLF_PATHS']=env['LTLF_PATHS']
sys.path.insert(0,str(ROOT/'replay'))
from artifact_paths import AUDIT_ROOT
commands=[]
for ds in ['coveo','electronics','cosmetics']:
    for condition in ['full_raw','short_raw','prefix_ltlf','counts','full_ltlf']:
        runner='run_coveo.py' if ds=='coveo' else 'run_act_rel_seed42.py'
        extra=['--protocol','protected_first_v1'] if ds=='coveo' else (['--observation','pre_purchase'] if ds=='cosmetics' else [])
        cmd=[sys.executable,str(ROOT/'replay'/runner),'--dataset',ds,'--seed','42','--condition',condition,'--max-minutes',str(args.max_minutes_per_run),*extra]
        prefix={'coveo':'protected_first_v1_seed42','electronics':'electronics_act_rel_seed42_v1','cosmetics':'cosmetics_pre_purchase_act_rel_seed42_v1'}[ds]
        run_name=prefix+'_'+condition
        commands.append((cmd,run_name))
        if args.initializations and condition in ['counts','full_ltlf']:
            commands.extend([(cmd+['--init-seed',str(i)],run_name+f'_init{i}') for i in [43,44]])
for cmd,run_name in commands:
    print(subprocess.list2cmdline(cmd),flush=True)
    if not args.dry_run:
        subprocess.run(cmd,env=env,check=True)
        if not (AUDIT_ROOT/'runs'/run_name/'metrics.json').exists():
            raise SystemExit(f'Run paused before completion: {run_name}. Resume this command with a larger per-run budget.')
print(f'{len(commands)} commands '+('listed.' if args.dry_run else 'completed.'))
