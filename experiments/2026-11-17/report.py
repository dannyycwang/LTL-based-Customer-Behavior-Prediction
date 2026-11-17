from common import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bootstrap_submission_controls import score_function
from threadpoolctl import threadpool_limits

NAMES={'raw':'Raw','counts':'Raw + Counts','transitions':'Raw + Counts + Transitions','ltlf':'Raw + Counts + LTLf','ltlf_frozen':'Raw + Counts + Learned LTLf','declare':'Raw + Counts + DECLARE','ltlf_matched_rank':'Raw + Counts + Re-ranked LTLf'}
def bootstrap(dataset,runs,out,label):
    preds={k:pd.read_csv(p/'predictions.csv.gz',dtype={'session_id':str,'user_id':str}) for k,p in runs.items()}
    ref=next(iter(preds.values()));y=ref.label.to_numpy()
    for p in preds.values():assert np.array_equal(ref.session_id,p.session_id) and np.array_equal(y,p.label)
    groups=pd.factorize(ref.user_id)[0] if 'user_id' in ref else np.arange(len(y));unit='user' if 'user_id' in ref else 'session'
    n=groups.max()+1;rng=np.random.default_rng(20261116);B=1500
    scorers={k:score_function(y,p.probability.to_numpy(),p.threshold.iloc[0]) for k,p in preds.items()}
    reps={k:np.empty((B,2)) for k in runs};point={k:f(np.ones(len(y),int)) for k,f in scorers.items()}
    for b in range(B):
        weight=np.bincount(rng.integers(n,size=n),minlength=n)[groups]
        for k,f in scorers.items():reps[k][b]=f(weight)
    np.savez_compressed(out/f'{dataset}_{label}_bootstrap.npz',**reps)
    logic='ltlf' if 'ltlf' in runs else 'ltlf_frozen';rows=[]
    for baseline in ['counts','transitions','declare']:
        if baseline not in runs:continue
        delta=reps[logic]-reps[baseline];ci=np.quantile(delta,[.025,.975],axis=0)
        for j,metric in enumerate(['MCC','AP']):rows.append(dict(dataset=dataset,setting=label,model_seed=42,first=logic,second=baseline,metric=metric,delta=point[logic][j]-point[baseline][j],lower95=ci[0,j],upper95=ci[1,j],bootstrap_unit=unit,n_units=int(n),replicates=B))
    return rows

def table_tex(df,path):
    df.to_latex(path,index=False,float_format=lambda v:f'{v:.4f}',escape=True)

def main(neural=False):
    out=HERE/'results';out.mkdir(exist_ok=True);rows=[];contrasts=[]
    base='neural' if neural else 'linear'
    for ds in ['Coveo','Electronics','Cosmetics']:
        runs={}
        conditions=['raw','counts','transitions','ltlf'] if neural else ['counts','transitions','declare','ltlf_frozen','ltlf_matched_rank']
        for c in conditions:
            for seed in ([42,43,44] if neural else [42]):
                p=HERE/base/(f'{ds}_{c}_{seed}' if neural else ds+'/'+c)
                if not (p/'metrics.json').exists():
                    if neural:raise RuntimeError(f'Missing formal run {p}')
                    continue
                r=json.loads((p/'metrics.json').read_text());r['seed']=seed;rows.append(r)
                if seed==42:runs[c]=p
        contrasts.extend(bootstrap(ds,runs,out,base))
    scores=pd.DataFrame(rows);scores.to_csv(out/f'{base}_metrics.csv',index=False)
    if neural:
        summary=scores.groupby(['dataset','condition'])[['MCC','AP']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
        assert (scores.groupby(['dataset','condition']).size()==3).all()
        summary.to_csv(out/'neural_mean_sample_sd.csv',index=False);table_tex(summary,out/'neural_mean_sample_sd.tex')
        seed_differences=[]
        for (dataset,seed),group in scores.groupby(['dataset','seed']):
            g=group.set_index('condition')
            for comparator in ['counts','transitions']:
                seed_differences.append(dict(dataset=dataset,seed=int(seed),comparator=comparator,
                    delta_MCC=g.loc['ltlf','MCC']-g.loc[comparator,'MCC'],delta_AP=g.loc['ltlf','AP']-g.loc[comparator,'AP']))
        sd=pd.DataFrame(seed_differences);sd.to_csv(out/'neural_seed_differences.csv',index=False)
        sd.groupby(['dataset','comparator'])[['delta_MCC','delta_AP']].agg(['mean','std']).to_csv(out/'neural_seed_difference_mean_sd.csv')
    else:table_tex(scores[['dataset','condition','MCC','AP']],out/'template_comparison.tex')
    diffs=pd.DataFrame(contrasts);diffs.to_csv(out/f'{base}_paired_differences.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    colors={'counts':'#0072B2','transitions':'#D55E00','declare':'#009E73'}
    for ax,metric in zip(axes,['MCC','AP']):
        d=diffs[diffs.metric==metric].reset_index(drop=True)
        for j,r in d.iterrows():ax.errorbar(r.delta,j,xerr=[[r.delta-r.lower95],[r.upper95-r.delta]],fmt='o',color=colors[r.second],capsize=3)
        ax.set_yticks(range(len(d)),[r.dataset+' vs '+r.second.title() for r in d.itertuples()]);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Learned LTLf minus comparator: '+metric);ax.grid(axis='x',alpha=.2)
    fig.suptitle(('Neural' if neural else 'Logistic regression')+' | event 10, H=10 | fixed seed 42 | paired 95% CI',fontsize=11)
    for ext in ['pdf','png']:fig.savefig(out/f'{base}_paired_differences.{ext}',dpi=220)
    plt.close(fig)
    print(base,'report complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--neural',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=1):main(args.neural)
