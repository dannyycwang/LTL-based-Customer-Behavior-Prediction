"""Aggregate the frozen pilot, produce a single principal table and paired AP figure."""
from pilot import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    metrics=pd.concat([pd.read_csv(HERE/d/'metrics.csv') for d in CFG['datasets']],ignore_index=True)
    diff=pd.concat([pd.read_csv(HERE/d/'paired_differences.csv') for d in CFG['datasets']],ignore_index=True)
    cohorts=pd.concat([pd.read_csv(HERE/d/'cohort_audit.csv') for d in CFG['datasets']],ignore_index=True)
    for col in cohorts.columns:
        if col!='users':cohorts[col]=cohorts[col].fillna(0)
    metrics.to_csv(HERE/'metrics.csv',index=False);diff.to_csv(HERE/'paired_differences.csv',index=False);cohorts.to_csv(HERE/'cohort_audit.csv',index=False)
    table=metrics[metrics.landmark==20].pivot(index=['dataset','condition','n_test','positives','features'],columns='metric',values='value').reset_index()
    ap=metrics[(metrics.landmark==20)&(metrics.metric=='AP')][['dataset','condition','lower_95','upper_95']]
    table=table.merge(ap,on=['dataset','condition']);table.to_csv(HERE/'main_results.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),sharex=True,sharey=True)
    colors={'C-A':'#136F63','C-B':'#4568B0','C-D':'#C16A23'}
    for ax,dataset in zip(axes,CFG['datasets']):
        part=diff[(diff.dataset==dataset)&(diff.metric=='AP')]
        for k,t in enumerate([30,20,10]):
            for i,c in enumerate(['C-A','C-B','C-D']):
                r=part[(part.landmark==t)&(part.contrast==c)].iloc[0];y=k+(i-1)*.20
                ax.plot([r.lower_95,r.upper_95],[y,y],color=colors[c],lw=2)
                ax.plot(r.delta,y,'o',color=colors[c],ms=6,label=c if k==0 else None)
        ax.axvline(0,color='#666666',lw=.8,ls='--');ax.axhspan(.65,1.35,color='#eef1f4',zorder=-1)
        ax.set_title(dataset);ax.set_yticks([0,1,2],['t = 30','t = 20 (primary)','t = 10'])
        ax.set_xlabel('Paired AP difference (95% percentile interval)');ax.grid(axis='x',alpha=.18)
        ax.set_xticks([-.010,-.005,0.,.005])
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(loc='best',frameon=False)
    fig.suptitle('Exploratory logistic regression pilot | m = 10, H = 10',fontsize=12)
    fig.text(.5,.015,'Fixed models/formulas; 1,500 paired resamples. Coveo: sessions. Cosmetics: users.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(HERE/'paired_ap_differences.png',dpi=180);fig.savefig(HERE/'paired_ap_differences.svg');plt.close(fig)
    timings=[]
    for ds in CFG['datasets']:
        for file in ['raw_audit.json','preparation.json','mining/manifest.json','fit_timing.json','evaluation_timing.json','boundary_followup.json']:
            p=HERE/ds/file
            if p.exists():
                obj=json.loads(p.read_text());timings.append(dict(dataset=ds,stage=file,seconds=obj.get('total_seconds',obj.get('seconds')),train_seconds=obj.get('train_seconds')))
    sleep=json.loads((HERE/'system_sleep.json').read_text()) if (HERE/'system_sleep.json').exists() else None
    for row in timings:
        row['known_sleep_seconds']=0.
        if sleep and row['dataset']=='Coveo' and row['stage']=='mining/manifest.json':
            row['known_sleep_seconds']=(pd.Timestamp(sleep['wake_utc'])-pd.Timestamp(sleep['sleep_utc'])).total_seconds()
        if sleep and row['dataset']=='Cosmetics' and row['stage']=='fit_timing.json':
            row['known_sleep_seconds']=sum((pd.Timestamp(s['wake_utc'])-pd.Timestamp(s['sleep_utc'])).total_seconds() for s in sleep.get('additional_sleep_intervals',[]) if 'current formal fit invocation' in s.get('affected_stage',''))
        row['wall_seconds_excluding_known_sleep']=None if row['seconds'] is None else row['seconds']-row['known_sleep_seconds']
        row['scope']='wall time, not CPU time; stages can overlap or nest; excludes unrecorded aborted work'
    pd.DataFrame(timings).to_csv(HERE/'runtime.csv',index=False)
    step_costs=[]
    for ds in CFG['datasets']:
        for t in CFG['landmarks']:
            for c in ('B','C','D'):
                if t==10 and c=='C':continue  # reused B, not a second executed selection
                p=HERE/ds/f't{t}'/c/'selection_steps.csv'
                if p.exists():
                    z=pd.read_csv(p)
                    suspended=0.
                    if sleep and ds=='Cosmetics' and t==10 and c=='D' and len(z)>=2:
                        suspended=sum((pd.Timestamp(s['wake_utc'])-pd.Timestamp(s['sleep_utc'])).total_seconds() for s in sleep.get('additional_sleep_intervals',[]) if 'Cosmetics t10 D step2' in s.get('affected_stage',''))
                    step_costs.append(dict(dataset=ds,landmark=t,condition=c,completed_steps=len(z),seconds=z.seconds.sum(),known_sleep_seconds=suspended,wall_seconds_excluding_known_sleep=z.seconds.sum()-suspended,scope='sum of measured completed steps across invocations; excludes baseline/preparation/aborted partial work; not CPU time'))
    pd.DataFrame(step_costs).to_csv(HERE/'selection_runtime.csv',index=False)
    selected=[]
    for ds in CFG['datasets']:
        for t in CFG['landmarks']:
            for c in ('B','C','D'):
                p=HERE/ds/f't{t}'/c/'selection.json';obj=json.loads(p.read_text())
                for step,r in enumerate(obj['selected'],1):selected.append(dict(dataset=ds,landmark=t,condition=c,step=step,formula=r['formula_str']))
    pd.DataFrame(selected).to_csv(HERE/'selected_formulas.csv',index=False)
    print(table.to_string(index=False));print(diff[diff.metric=='AP'].to_string(index=False))

if __name__=='__main__':main()
