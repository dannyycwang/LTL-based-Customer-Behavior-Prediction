from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
# Requires numpy, pandas, and matplotlib.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
AUD=ROOT/'results';W=ROOT
FIG=W/'generated/figures';FIG.mkdir(parents=True,exist_ok=True)
TAB=W/'generated/tables';TAB.mkdir(parents=True,exist_ok=True)
GSTAT=W/'generated/statistics';GSTAT.mkdir(parents=True,exist_ok=True)
E=pd.read_csv(W/'statistics/estimates.csv');D=pd.read_csv(W/'statistics/contrasts.csv')
DS=['Coveo','Electronics','Cosmetics']
COL={'Full raw':'#34495E','Counts':'#BD822F','LTLf':'#008C83','Transitions':'#3E75B0','Both':'#986896','Prefix':'#91A0B0'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'lines.linewidth':1.2})
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',pad_inches=.035,dpi=350)
    plt.close(fig)
def ase(r,bold=False):
    v=f'{r.estimate:.3f}';se=f'{r.bootstrap_se:.3f}'
    return ('\\bse' if bold else '\\vse')+'{'+v+'}{'+se+'}'
def find(d,setting,c,m):return E[(E.dataset==d)&(E.setting==setting)&(E.condition==c)&(E.metric==m)].iloc[0]
def tab(name,cols,lines):
    text='\\begin{revblock}\n\\begin{tabular}{@{}'+cols+'@{}}\n'+ '\n'.join(lines)+'\n\\end{tabular}\n\\end{revblock}\n'
    (TAB/(name+'.tex')).write_text(text,encoding='utf-8')
header=[r'\toprule',r'& \multicolumn{2}{c}{Coveo} & \multicolumn{2}{c}{Electronics} & \multicolumn{2}{c}{Cosmetics} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',r'Input & MCC & AP & MCC & AP & MCC & AP \\',r'\midrule']
neural=['full_raw','short_raw','prefix_ltlf','counts','full_ltlf'];names=['Full raw','Short raw','Short + Prefix-LTLf','Short + Counts','Short + Full-LTLf']
lines=header.copy()
for c,label in zip(neural,names):
    row=[label]
    for ds in DS:
        for m in ['MCC','AP']:
            r=find(ds,'neural_all',c,m);best=E[(E.dataset==ds)&(E.setting=='neural_all')&(E.metric==m)].estimate.max()
            row.append(ase(r,r.estimate==best))
    lines.append(' & '.join(row)+r' \\')
lines.append(r'\bottomrule');tab('neural','lrrrrrr',lines)

linear=['Short raw','Short + Counts','Short + Counts + LTLf','Short + Counts + transitions','Short + Counts + transitions + LTLf']
lnames=['Raw prefix','+ Counts','+ Counts + LTLf','+ Counts + Transitions','+ Counts + Both']
lines=header.copy()
for setting,title in [('full_observation','(a) Full-observation summaries'),('next_10','(b) Event-10 prediction, next 10 events')]:
    lines.append(r'\multicolumn{7}{@{}l}{\textit{'+title+r'}} \\')
    for c,label in zip(linear,lnames):
        row=[label]
        for ds in DS:
            for m in ['MCC','AP']:
                r=find(ds,setting,c,m);best=E[(E.dataset==ds)&(E.setting==setting)&(E.metric==m)].estimate.max()
                row.append(ase(r,r.estimate==best))
        lines.append(' & '.join(row)+r' \\')
    lines.append(r'\midrule' if setting=='full_observation' else r'\bottomrule')
tab('linear','lrrrrrr',lines)

co=pd.read_csv(AUD/'submission_iterations_seed42/cohort_counts.csv')
lines=[r'\toprule',r'Dataset & Full observation: train / val. / test & At event 10: train / val. / test & $\pi_{\rm full}$ / $\pi_{10}$ & Raw cap \\',r'\midrule']
for d,cap in zip(DS,[200,512,200]):
    def fmt(p):return ' / '.join(f'{int(co[(co.dataset==d)&(co.protocol==p)&(co.split==s)].iloc[0]["n"]):,}' for s in ['train','val','test'])
    a=co[(co.dataset==d)&(co.protocol=='full_observation')&(co.split=='test')].iloc[0]
    b=co[(co.dataset==d)&(co.protocol=='next_10')&(co.split=='test')].iloc[0]
    lines.append(f'{d} & {fmt("full_observation")} & {fmt("next_10")} & {a.prevalence*100:.1f} / {b.prevalence*100:.1f} & {cap}'+r' \\')
lines.append(r'\bottomrule');tab('cohorts','lrrrc',lines)

rt=pd.read_csv(AUD/'submission_controls_seed42/inference_runtime_repetitions.csv')
train=pd.read_csv(AUD/'submission_controls_seed42/training_runtime.csv')
lines=[r'\toprule',r'Dataset & Neural input & MCC & Train + val. s (epochs) & Test cohort s & CUDA MiB \\',r'\midrule']
for d in DS:
    for j,(c,name) in enumerate(zip(['Full raw','Short + Counts','Short + Full-LTLf'],['Full raw','+ Counts','+ Full-LTLf'])):
        t=train[(train.dataset==d)&(train.condition==c)].iloc[0];r=rt[(rt.dataset==d)&(rt.condition==c)]
        lines.append(' & '.join([d if j==0 else '',name,f'{t.MCC:.3f}',f'{t.training_and_validation_sec:.1f} ({int(t.completed_epochs)})',r'\vse{'+f'{r.total_sec.mean():.2f}'+'}{'+f'{r.total_sec.std(ddof=1):.2f}'+'}',f'{r.peak_cuda_allocated_mib.mean():.1f}'])+r' \\')
    lines.append(r'\midrule' if d!=DS[-1] else r'\bottomrule')
tab('cost','llrrrr',lines)

# RQ1: compare the scope change where additional events exist and where they do not.
fig,ax=plt.subplots(figsize=(3.35,1.8))
for i,d in enumerate(DS):
    for setting,off,color,marker in [('neural_short',.11,'#9CA9B5','s'),('neural_long',-.11,COL['LTLf'],'o')]:
        r=D[(D.dataset==d)&(D.setting==setting)&(D['first']=='full_ltlf')&(D.second=='prefix_ltlf')&(D.metric=='MCC')].iloc[0]
        y=2-i+off
        ax.plot([0,r.estimate],[y,y],color=color,lw=1.3,alpha=.6)
        ax.errorbar(r.estimate,y,xerr=r.bootstrap_se,fmt=marker,ms=4,color=color,capsize=2,elinewidth=.9,zorder=3)
    long=D[(D.dataset==d)&(D.setting=='neural_long')&(D.metric=='MCC')].iloc[0]
    total=find(d,'neural_all','full_raw','MCC')['n']
    ax.text(.525,2-i,f'{int(long["n"]):,}\n({long["n"]/total:.1%})',fontsize=6.8,va='center',color='#526374')
ax.text(.525,2.53,'L > 10',fontsize=7,color='#526374')
ax.set(yticks=[2,1,0],yticklabels=DS,xlim=(-.025,.66),ylim=(-.42,2.75),xticks=[0,.1,.2,.3,.4,.5],xlabel='ΔMCC: Full-LTLf − Prefix-LTLf')
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.18);ax.axvline(0,color='#B2BDC8',lw=.7)
handles=[Line2D([0],[0],marker='s',lw=0,color='#9CA9B5',label='L ≤ 10'),Line2D([0],[0],marker='o',lw=0,color=COL['LTLf'],label='L > 10')]
ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,1.06),ncols=2,frameon=False,handletextpad=.3,columnspacing=1)
fig.subplots_adjust(left=.22,right=.99,bottom=.27,top=.90);save(fig,'rq1_scope')

# RQ3: paired incremental value at each future horizon, rather than redundant levels.
fig,axes=plt.subplots(1,3,figsize=(7.05,1.85),sharey=True)
for ax,d in zip(axes,DS):
    for name,c,idx,marker in [('LTLf',COL['LTLf'],2,'o'),('Transitions',COL['Transitions'],3,'s'),('Both',COL['Both'],4,'^')]:
        ys=[];ses=[]
        for h in [5,10,20]:
            v=np.load(W/f'statistics/{d.lower()}_next_{h}_replicates.npz')
            delta=v['values'][:,idx,0]-v['values'][:,1,0]
            ys.append(v['point'][idx,0]-v['point'][1,0]);ses.append(delta.std(ddof=1))
        ax.errorbar([5,10,20],ys,yerr=ses,marker=marker,color=c,label=name,ms=3,capsize=2,elinewidth=.7,lw=1)
    n=int(find(d,'next_10',linear[0],'MCC')['n'])
    ax.set_title(f'{d}  (n={n:,})',fontsize=8.2,pad=5)
    ax.set(xticks=[5,10,20],ylim=(-.012,.16),yticks=[0,.05,.10,.15],xlabel='Future events H')
    ax.axhline(0,color='#657689',lw=.65,ls='--');ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('ΔMCC over Raw + Counts')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.55,1.12),ncols=3,frameon=False,columnspacing=2,handlelength=1.3)
fig.subplots_adjust(left=.08,right=.995,bottom=.27,top=.82,wspace=.19);save(fig,'rq3_horizon')

# RQ4: three measured boundaries, not an undifferentiated speedup claim.
up=pd.read_csv(AUD/'submission_controls_seed42/coveo_update_timing_repetitions.csv')
methods=['Full raw','Short + Counts','Short + Full-LTLf'];short=['Full raw','Counts','LTLf']
fig,axes=plt.subplots(1,3,figsize=(7.05,2.05),gridspec_kw={'width_ratios':[1.05,1.1,1.]})
ax=axes[0]
for i,(c,label) in enumerate(zip(methods,short)):
    a=up[up.condition==c].milliseconds_per_update.to_numpy();y=2-i
    ax.plot(a,[y]*3,'o',ms=2.8,alpha=.45,color=COL[label])
    ax.errorbar(a.mean(),y,xerr=a.std(ddof=1),fmt='D',ms=4,color=COL[label],capsize=3,elinewidth=1)
    ax.text(0,y+.20,f'{a.mean():.1f} ± {a.std(ddof=1):.1f}',fontsize=6.9,color=COL[label])
ax.set(yticks=[2,1,0],yticklabels=short,xlim=(-3,120),ylim=(-.5,2.55),xticks=[0,40,80,120],xlabel='ms / neural update')
ax.set_title('(a) Controlled updates · Coveo',loc='left',fontsize=8,pad=9)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.18)
ax=axes[1];bottom=np.zeros(3)
parts=['raw_encoding_sec','symbolic_feature_sec','inference_and_transfers_sec'];pc=['#ABBAD0','#D7B675','#85B6A9'];pl=['Raw packing','Summary CPU','Model + transfers']
for col,color,label in zip(parts,pc,pl):
    val=np.array([rt[(rt.dataset=='Coveo')&(rt.condition==c)][col].mean() for c in methods]);ax.bar(np.arange(3),val,bottom=bottom,width=.57,color=color,label=label,edgecolor='white',linewidth=.5);bottom+=val
sd=np.array([rt[(rt.dataset=='Coveo')&(rt.condition==c)].total_sec.std(ddof=1) for c in methods]);ax.errorbar(np.arange(3),bottom,yerr=sd,fmt='none',color='#334155',capsize=2,elinewidth=.7)
for i,v in enumerate(bottom):ax.text(i,v+.24,f'{v:.2f}',ha='center',fontsize=7)
ax.set(xticks=range(3),xticklabels=short,ylim=(0,5.5),yticks=[0,2,4],ylabel='seconds / test cohort')
ax.set_title('(b) Feature-inclusive inference',loc='left',fontsize=8,pad=9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(-.06,-.25),fontsize=6.5,frameon=False,ncols=1,handlelength=.9,labelspacing=.25)
ax=axes[2]
for i,d in enumerate(DS):
    vals=[rt[(rt.dataset==d)&(rt.condition==c)].peak_cuda_allocated_mib.mean() for c in methods]
    ax.plot([min(vals),max(vals)],[2-i]*2,color='#C8D0D8',lw=2)
    for j,label in enumerate(short):ax.scatter(vals[j],2-i+(j-1)*.07,color=COL[label],marker=['s','o','D'][j],s=19,zorder=3)
    ax.text(vals[0],2-i+.18,f'{vals[0]:.0f}',ha='center',fontsize=6.8,color=COL['Full raw'])
    ax.text(vals[2],2-i-.27,f'{vals[2]:.0f}',ha='center',fontsize=6.8,color=COL['LTLf'])
ax.set(xscale='log',xlim=(15,3500),ylim=(-.5,2.55),yticks=[2,1,0],yticklabels=['Coveo','Elec.','Cosm.'],xticks=[20,100,500,2000],xticklabels=['20','100','500','2000'],xlabel='peak CUDA MiB (log scale)')
ax.set_title('(c) Inference GPU allocation',loc='left',fontsize=8,pad=9);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.16)
fig.subplots_adjust(left=.075,right=.99,top=.83,bottom=.29,wspace=.54);save(fig,'rq4_cost')

# Machine-readable measurements underlying the updated mean ± SD reporting.
stats=rt.groupby(['dataset','condition']).agg(total_mean=('total_sec','mean'),total_sd=('total_sec','std'),feature_mean=('symbolic_feature_sec','mean'),peak_mib=('peak_cuda_allocated_mib','mean')).reset_index()
stats.to_csv(GSTAT/'runtime_mean_sd.csv',index=False)
up.groupby('condition').milliseconds_per_update.agg(['mean','std']).to_csv(GSTAT/'update_mean_sd.csv')
print('Compact tables and RQ figures generated.')
