"""Publication-oriented paired contrasts, with conditional 95% intervals."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent
d=pd.read_csv(p/'results_uncertainty.csv')
order=[(ds,mode) for ds in ['Coveo','Electronics','Cosmetics'] for mode in ['full_observation','next_10']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True,layout='constrained')
for axis,metric in zip(axes,['MCC','AP']):
    for second,color,offset,label in [('A','#167A9B',-.13,'Residual minus Counts + Transitions'),
                                      ('B','#C56A22',.13,'Residual minus Original Top-10')]:
        for i,(dataset,protocol) in enumerate(order):
            row=d[(d.dataset==dataset)&(d.protocol==protocol)&(d['first']=='C')&(d.second==second)&(d.metric==metric)].iloc[0]
            axis.errorbar(row.estimate,i+offset,xerr=[[max(0,row.estimate-row.lower95)],
                [max(0,row.upper95-row.estimate)]],fmt='o',color=color,capsize=3,markersize=5,
                label=label if i==0 else None)
    axis.axvline(0,color='#555555',linewidth=.8)
    axis.grid(axis='x',alpha=.15);axis.set_xlabel('Paired change in '+metric)
axes[0].set_yticks(range(len(order)),[ds+' / '+('Full' if mode=='full_observation' else 'Event 10') for ds,mode in order])
axes[0].invert_yaxis()
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',frameon=False,ncols=1)
fig.suptitle('Residual-guided formula selection: paired 95% bootstrap intervals',fontsize=12)
fig.savefig(p/'paired_contrasts.png',dpi=300)
fig.savefig(p/'paired_contrasts.svg')
