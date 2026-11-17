"""Generate every figure/table from saved audit/diagnostic outputs, without invented scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
D=ROOT/'coveo_candidate_diagnostics'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'

def save(fig,name):
    fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(ROOT/(name+'.png'),dpi=320,bbox_inches='tight');plt.close(fig)

def latex(df,path,caption):
    def esc(x):return str(x).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&').replace('<','\\textless{}').replace('>','\\textgreater{}')
    text='% '+caption+'\n\\begin{tabular}{'+'l'*len(df.columns)+'}\n\\hline\n'
    text+=' & '.join(esc(x) for x in df.columns)+' \\\\\n\\hline\n'
    for _,r in df.iterrows():text+=' & '.join(esc(x) for x in r)+' \\\\\n'
    text+='\\hline\n\\end{tabular}\n';path.write_text(text,encoding='utf-8')

def main():
    meta=json.loads((D/'metadata.json').read_text());changes=pd.read_csv(D/'prefix_full_formula_changes.csv')
    long=changes[changes.stratum=='L>10'];sh=pd.read_csv(D/'shuffle_changes.csv')
    agg=sh.groupby('formula_id',sort=False).agg(mean_changed_rate=('changed_rate','mean'),sd_across_shuffle=('changed_rate','std')).reset_index()
    agg.to_csv(D/'shuffle_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'wspace':.3})
    x=np.arange(len(long));axes[0].bar(x-.18,100*long.zero_to_one_rate,.36,color=BLUE,label='0 to 1');axes[0].bar(x+.18,100*long.one_to_zero_rate,.36,color=ORANGE,label='1 to 0')
    axes[0].set(xticks=x,xticklabels=long.formula_id,ylabel='Sessions with changed bit (%)',xlabel='Fixed formula ID',title=f'Prefix 10 vs full trace (L>10; n={meta["test_long_n"]:,})');axes[0].legend(frameon=False)
    a=agg.set_index('formula_id').loc[long.formula_id]
    axes[1].bar(x,100*a.mean_changed_rate,color=GREEN,yerr=100*a.sd_across_shuffle,capsize=3)
    axes[1].set(xticks=x,xticklabels=long.formula_id,ylabel='Sessions with changed bit (%)',xlabel='Fixed formula ID',title='Symbolic event shuffling vs original full trace')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Coveo candidate CSV: fixed seed-42 Act+Rel basis (K=10)',y=1.02,fontsize=13)
    fig.text(.5,-.07,f'Candidate test partition: n={meta["test_n"]:,}; identity with paper splits not established.\nRight: mean +/- SD over 10 shuffle seeds; symbolic co-occurrences and proposition counts preserved. No predictor retraining.',ha='center',fontsize=9)
    save(fig,'coveo_formula_diagnostics')
    example=json.loads((D/'example.json').read_text());tr=example['trace'];n=len(tr)
    used=['b','d','a','r','fp','rp','sp','pc','p2','p3','sc','cc','rc']
    props=[p for p in used if any(p in s for s in tr)]
    matrix=np.array([[p in s for s in tr] for p in props],dtype=int)
    fig,ax=plt.subplots(figsize=(10,5.5));fig.subplots_adjust(bottom=.30)
    ax.imshow(matrix,cmap=matplotlib.colors.ListedColormap(['#F2F2F2',BLUE]),vmin=0,vmax=1,aspect='auto')
    ax.axvline(9.5,color=ORANGE,linewidth=2,linestyle='--');ax.set(xticks=range(n),xticklabels=range(1,n+1),yticks=range(len(props)),yticklabels=props,xlabel='Event position',ylabel='Proposition')
    ax.set_title(f'Real candidate trace example: {example["formula"]["formula_id"]}, prefix={example["prefix_value"]}, full={example["full_value"]}')
    fig.text(.5,.04,example['formula']['formula_str']+'\nSelection: shortest changed test trace with L>10; ties by session ID; first changed formula.\n'+f'Session {example["session_id"][:16]}...; illustrative diagnostic, not evidence of prediction improvement.',ha='center',fontsize=9,linespacing=1.6)
    save(fig,'coveo_real_trace_example')
    overlap=pd.read_csv(ROOT/'coveo_formula_training_overlap.csv');te=overlap[overlap.downstream_split=='test']
    fig,ax=plt.subplots(figsize=(7.5,5));fig.subplots_adjust(bottom=.25)
    bars=ax.bar(te.seed.astype(str),100*te.fraction_of_downstream_split,color=ORANGE,width=.58)
    ax.bar_label(bars,labels=[f'{n:,} sessions\n{r:.2%}' for n,r in zip(te.formula_training_overlap_n,te.fraction_of_downstream_split)],padding=5)
    ax.set(ylabel='Reconstructed downstream test set (%)',xlabel='Downstream seed',ylim=(0,10.5),title='Coveo: formula-training sessions entering downstream test')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.04,'20,000 formula-training traces matched stored Drive JSON in order.\nDownstream splits reconstructed from supplied notebook; original saved ID lists still needed.',ha='center',fontsize=9,linespacing=1.6)
    save(fig,'coveo_split_overlap')
    historical=pd.read_csv(ROOT/'table4_correspondence.csv');rows=[]
    for _,r in historical.iterrows():
        row={'Dataset':r.dataset,'Scheme':r.scheme,'Condition':r.condition}
        for m in ['f1','mcc','auprc','brier']:row[m.upper()]=f'{r[m+"_mean"]:.3f} $\\pm$ {r[m+"_sd"]:.3f}'
        rows.append(row)
    # Preserve LaTeX math for the mean/SD cells.
    hist=pd.DataFrame(rows);latex(hist,ROOT/'historical_table4.tex','Historical values from saved runs; mean and sample SD (ddof=1); provenance conflicts unresolved; not new experiments.')
    p=ROOT/'historical_table4.tex';p.write_text(p.read_text().replace('\\textbackslash{}pm','\\pm'),encoding='utf-8')
    latex(pd.read_csv(D/'length_strata.csv'),D/'length_strata.tex','Candidate data strata, not paper seed-specific test strata.')
    for src in ['prefix_full_formula_changes','shuffle_summary']:
        latex(pd.read_csv(D/(src+'.csv')).round(6),D/(src+'.tex'),'Diagnostic only; source data and seeds in metadata.json.')
    print('Generated 3 PDF/PNG figures and LaTeX tables from saved outputs.')

if __name__=='__main__':main()
