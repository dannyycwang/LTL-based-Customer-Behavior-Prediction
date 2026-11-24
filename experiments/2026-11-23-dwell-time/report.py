from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def semantic(t):
    if isinstance(t,str):return {'dl':'low known outgoing gap','dm':'medium known outgoing gap','dh':'high known outgoing gap','du':'unknown outgoing gap','a':'add-to-cart','b':'browse','d':'detail/view','r':'remove-from-cart','true':'true','false':'false','fp':'first observed occurrence of this product','rp':'previously observed product','sp':'same product as the previous product reference','pc':'product differs from the previous product reference','p2':'this product has appeared at least twice','p3':'this product has appeared at least three times','dp1':'one distinct known product observed so far','dp2':'two distinct known products observed so far','dp3':'at least three distinct known products observed so far','hp':'known product present','np':'known product absent','hc':'known category present','sc':'same category as the previous category reference','cc':'category differs from the previous category reference','rc':'previously observed category'}.get(t,t)
    op=t[0];a=semantic(t[1])
    if op in ['!','not']:return f'not ({a})'
    if op=='X':return f'at the next observed event ({a})'
    if op=='F':return f'at some observed event ({a})'
    if op=='G':return f'at every observed event ({a})'
    b=semantic(t[2])
    return {'U':f'({a}) holds until ({b}), which must occur','&':f'({a}) and ({b})','&&':f'({a}) and ({b})','|':f'({a}) or ({b})','||':f'({a}) or ({b})','->':f'if ({a}), then ({b})'}[op]

def tokens(t):
    if isinstance(t,str):return {t}
    return set().union({t[0]},*(tokens(x) for x in t[1:]))

def formulas():
    records=[]
    for ds in CFG['datasets']:
        d=load(privdir(ds)/'prepared.pkl')['train'];bundle=load(privdir(ds)/'linear_frozen.pkl');prep=json.loads((outdir(ds)/'preparation.json').read_text())
        for c in ['M1','M3','M4','M5']:
            f=bundle['conditions'][c];trees=[freeze(f['rows'][j]['formula_tree']) for j in f['selected']]
            if not trees:continue
            scope='dwell' if c in ['M4','M5'] else 'full';props=prep['props']+(DWELL if scope=='dwell' else [])
            vals=oldfit.bits(d[scope],props,trees)
            for k,(j,t) in enumerate(zip(f['selected'],trees)):
                v=vals[:,k];matches=[];best=0.;best_name=None
                for tag in DWELL:
                    col=d['time'][:,prep['time_names'].index(tag+'_count')]
                    for cut in np.unique(col):
                        for positive in [True,False]:
                            pred=(col>=cut) if positive else (col<cut);agreement=float(np.mean(pred==v))
                            name=f'{tag}_count {">=" if positive else "<"} {cut:g}'
                            if agreement>best:best=agreement;best_name=name
                            if agreement==1:matches.append(name)
                tok=tokens(t);records.append(dict(dataset=ds,condition=c,rank=k+1,candidate=j,formula=repr(t),contains_dwell=bool(tok&set(DWELL)),contains_known_duration_bucket=bool(tok&{'dl','dm','dh'}),contains_unknown_marker='du' in tok,temporal_syntax=bool(tok&{'X','F','G','U'}),training_support=float(v.mean()),positive_support=float(v[d['label']==1].mean()),negative_support=float(v[d['label']==0].mean()),exact_training_dwell_count_redundancy='; '.join(matches),best_univariate_count_agreement=best,best_name=best_name,semantics=semantic(t)))
    pd.DataFrame(records).to_csv(HERE/'selected_formulas.csv',index=False)
    return pd.DataFrame(records)

def interval(row):return f"{row.delta:+.5f} [{row.lower_95:+.5f}, {row.upper_95:+.5f}]"
def interpret(row):
    if row.lower_95>0:return '固定模型的 AP 差值區間完全在零以上，但仍是探索性證據'
    if row.upper_95<0:return '固定模型的 AP 差值區間完全在零以下，未支持增益'
    return 'AP 差值區間跨零，未證實額外增益；不代表等價'

def main():
    metrics=pd.concat([pd.read_csv(outdir(ds)/'metrics.csv') for ds in CFG['datasets']],ignore_index=True)
    pairs=pd.concat([pd.read_csv(outdir(ds)/'paired_differences.csv') for ds in CFG['datasets']],ignore_index=True)
    metrics.to_csv(HERE/'metrics.csv',index=False);pairs.to_csv(HERE/'paired_differences.csv',index=False)
    nn=metrics[metrics.model=='NN'].groupby(['dataset','condition','metric']).value.agg(['mean','std']).reset_index();nn.to_csv(HERE/'neural_mean_sd.csv',index=False)
    paired_seed_summary=pairs[pairs.model=='NN'].groupby(['dataset','contrast','metric']).delta.agg(['mean','std']).reset_index()
    paired_seed_summary.to_csv(HERE/'neural_paired_mean_sd.csv',index=False)
    nr=[]
    for ds in CFG['datasets']:
        for c in CFG['neural_conditions']:
            for seed in CFG['neural_seeds']:
                root=outdir(ds)/f'{c}_{seed}';f=json.loads((root/'freeze.json').read_text());cf=json.loads((root/'config.json').read_text());ep=json.loads((root/'epochs.json').read_text())
                scores=metrics[(metrics.dataset==ds)&(metrics.model=='NN')&(metrics.condition==c)&(metrics.seed==seed)]
                nr.append(dict(dataset=ds,condition=c,seed=seed,parameters=f['parameters'],summary_dimension=cf['summary_dim'],epochs=f['epochs'],best_epoch=f['best_epoch'],threshold=f['threshold'],training_validation_seconds=sum(e['seconds'] for e in ep),**dict(zip(scores.metric,scores.value))))
    pd.DataFrame(nr).to_csv(HERE/'neural_runs.csv',index=False)
    main=metrics[metrics.model=='LR'].pivot(index=['dataset','condition'],columns='metric',values='value').reset_index()
    counts={ds:{c:len(f['selected']) for c,f in load(privdir(ds)/'linear_frozen.pkl')['conditions'].items()} for ds in CFG['datasets']}
    main['selected_logic_features']=[counts[r.dataset][r.condition] for _,r in main.iterrows()]
    main.to_csv(HERE/'main_results.csv',index=False)
    fs=formulas();cost=[];search=[]
    for ds in CFG['datasets']:
        cost.extend([dict(dataset=ds,**r) for r in json.loads((outdir(ds)/'runtime.json').read_text())])
        for c in ['M3','M4','M5']:
            s=json.loads((outdir(ds)/c/'selection.json').read_text());steps=pd.read_csv(outdir(ds)/c/'selection_steps.csv') if (outdir(ds)/c/'selection_steps.csv').exists() else pd.DataFrame()
            search.append(dict(dataset=ds,condition=c,initial=s['initial_candidate_count'],eligible=s['eligible_count'],selected=len(s['indices']),selection_seconds=steps.seconds.sum() if len(steps) else 0,candidate_refits=int(steps.candidate_count.sum()) if len(steps) else 0))
    pd.DataFrame(cost).to_csv(HERE/'runtime.csv',index=False);pd.DataFrame(search).to_csv(HERE/'selection_cost.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5));p=pairs[(pairs.metric=='AP')&(pairs.contrast=='M4-M3')&pairs.model.isin(['LR','NN_mean'])].reset_index(drop=True)
    for i,r in p.iterrows():ax.errorbar(r.delta,i,xerr=[[r.delta-r.lower_95],[r.upper_95-r.delta]],fmt='o',capsize=4,color='#126a89' if r.model=='LR' else '#b06420')
    ax.set_yticks(range(len(p)),[f'{r.dataset} — {r.model}' for _,r in p.iterrows()]);ax.axvline(0,color='gray',lw=1);ax.set_xlabel('Paired AP difference: M4 − M3 (95% percentile interval)');ax.set_title('t=20, m=10, H=10 · exploratory');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(HERE/'paired_ap_differences.png',dpi=200);fig.savefig(HERE/'paired_ap_differences.svg');plt.close(fig)
    zh=['# 固定 dwell 補充實驗結果','', '本輪固定矩陣已完成：兩資料集六個線性條件，加上 M2/M3/M4 各三個神經 optimization seeds，共18個神經 fits。舊 test 曾參與研究開發，所有結論均為 exploratory。時間變數是已完成 event gap，不是真實注意力時間。','', '## 直接回答研究問題','']
    en=['# Manuscript-ready exploratory results','', 'We evaluated a fixed event-20 cohort with a ten-event raw window and a ten-event first-purchase horizon on Coveo and Cosmetics. Electronics was included only in the historical audit. Outgoing event gaps were constructed after truncating the observable prefix; the terminal gap was explicitly unknown. Bucket thresholds and numeric transformations were fitted on training data only.','']
    for ds in CFG['datasets']:
        zh.extend([f'### {ds}',''])
        for ct,label in [('M2-M0','一般時間特徵'),('M4-M3','含 dwell vocabulary 的 learned LTLf 相對 Act+Rel LTLf'),('M4-M2','已提供普通時間特徵後的 LTLf 增益'),('M4-M5','learned LTLf 相對 DECLARE')]:
            r=pairs[(pairs.dataset==ds)&(pairs.model=='LR')&(pairs.metric=='AP')&(pairs.contrast==ct)].iloc[0]
            zh.append(f'- {label}（{ct}）：{interval(r)}；{interpret(r)}。')
            en.append(f'For {ds}, the linear {ct} AP difference was {interval(r)}.')
        r=pairs[(pairs.dataset==ds)&(pairs.model=='NN_mean')&(pairs.metric=='AP')&(pairs.contrast=='M4-M3')].iloc[0]
        zh.extend([f'- 神經模型主要對比（三 seed 平均 M4−M3）：{interval(r)}；{interpret(r)}。各 seed 與 mean/sample SD 見附表，三 seed 不當成三倍樣本。',''])
        en.append(f'The neural mean paired AP difference (M4−M3) across optimization seeds 42–44 was {interval(r)}; all seeds shared the same cluster resamples.')
        lr=pairs[(pairs.dataset==ds)&(pairs.model=='LR')&(pairs.metric=='AP')&(pairs.contrast=='M4-M3')].iloc[0]
        if lr.lower_95>0 and r.lower_95>0:agreement='線性與神經主要對比都支持此固定設定下的正向 AP 差值；仍不能推廣為普遍效益。'
        elif lr.delta*r.delta<0:agreement='線性與神經主要對比的點估計方向不同，不支持跨模型一致的增益主張。'
        else:agreement='至少一種模型的主要對比未證實正向增益，不能宣稱神經結果確認了線性模型的收益。'
        zh.extend([agreement,''])
        for ct in ['M4-M2','M3-M2']:
            rr=pairs[(pairs.dataset==ds)&(pairs.model=='NN_mean')&(pairs.metric=='AP')&(pairs.contrast==ct)].iloc[0]
            zh.append(f'- 神經三 seed 平均 {ct}：{interval(rr)}；{interpret(rr)}。')
            en.append(f'The neural seed-average {ct} AP difference was {interval(rr)}.')
        zh.extend(['','主要對比 M4−M3 的次要指標如下（MCC 越高越好，Brier 越低越好；不依方向挑選報告）：',''])
        for model,label in [('LR','線性'),('NN_mean','神經三 seed 平均')]:
            secondary=[]
            for metric in ['MCC','Brier']:
                rr=pairs[(pairs.dataset==ds)&(pairs.model==model)&(pairs.metric==metric)&(pairs.contrast=='M4-M3')].iloc[0]
                secondary.append(f'{metric} {interval(rr)}')
            zh.append(f'- {label}：'+ '；'.join(secondary)+'。')
            en.append(f'For the {model} primary contrast, the paired secondary differences were '+ '; '.join(secondary)+'. Higher MCC and lower Brier are preferable.')
        sub=fs[(fs.dataset==ds)&(fs.condition=='M4')];zh.append(f'M4 選中 {len(sub)} 個公式，其中 {int(sub.contains_dwell.sum()) if len(sub) else 0} 個含時間 vocabulary，{int(sub.contains_known_duration_bucket.sum()) if len(sub) else 0} 個含真正已知間隔 buckets，{int(sub.contains_unknown_marker.sum()) if len(sub) else 0} 個含未知 marker du，{int(sub.temporal_syntax.sum()) if len(sub) else 0} 個含 temporal operator 語法。du 在本資料也是末端標記，不能將只用 du 的公式當成 duration 效應。語法不等於不可替代的時序資訊；training count-rule 冗餘和 support 見 selected_formulas.csv。')
    zh.extend(['','## 主張與引用範圍','', '保留：在此固定 cohort、觀察邊界和模型下觀察到的 AP/MCC/Brier 數值與配對差值；邏輯特徵可被明確檢查。縮小：任何收益只能歸於整套 vocabulary／學習／選擇流程，不能歸因單一 dwell operator。神經結論限既有 encoder/fusion 與 LR 移用公式；不同模型結果不得互相替代。刪除或避免：真實注意力、因果機制、普遍 superiority、未顯著即等價、整體訓練加速，以及以舊 session-purchase 成績支持新固定 landmark 任務的主張。','', '舊三資料集 dwell 結果的可引用性見 inventory_zh.md。Electronics 原版確定洩漏；修補版仍有 master thresholds／未知零值混同限制。Coveo 舊 short-window 邊界與 median 填補不同；Cosmetics 舊 parser／purchase-relative task 不同。歷史彙總可註明限制後引用為背景，不能與本輪兩資料集新結果配對。','', '## 完整結果與限制','', main.to_markdown(index=False),'', 'AP 為 primary；不得忽略 metrics.csv 中的 MCC 與 Brier，完整 paired intervals 在 paired_differences.csv。所有 interval 固定模型、公式及 thresholds，不涵蓋 mining uncertainty；沒有多重比較校正。Cosmetics 按 user，Coveo 按 session，後者不能排除同人跨 session 相關性。樣本為歷史 cohort，非自然部署 prevalence。','', '時間成本分項見 runtime.csv／selection_cost.csv；selection 含候選分類器重擬合，不重複相加；cached-input neural training 不是端到端速度。所有個體資料、IDs、預測與模型在 private。原論文與既有實驗未覆寫。本輪矩陣完成後停止，未自動延伸研究。'])
    en.extend(['','The primary comparison contrasts two vocabulary and formula-learning pipelines, not the causal effect of a single dwell operator. All comparisons are exploratory because the test set previously participated in research development. Intervals condition on fitted models, formulas, and thresholds and exclude mining uncertainty. Failure to exclude zero is not evidence of equivalence. Formulas selected with logistic regression were transferred unchanged to the neural models, limiting architecture-level conclusions.','',main.to_markdown(index=False),'','Neural performance (mean and sample SD across three optimization seeds):','',nn.to_markdown(index=False),'','Figure caption. Paired AP differences between Act+Rel+Dwell LTLf (M4) and Act+Rel LTLf (M3), conditional on the same raw, strong-summary, and nonlogical timing inputs. Bars show 95% percentile intervals from 1,500 shared cluster resamples (Cosmetics: user; Coveo: session). Neural points average paired differences over three optimization seeds using identical cluster resamples across seeds. Models, formulas, and thresholds are fixed. Only Coveo and Cosmetics are new experiments; the three-dataset historical dwell results are audited separately.'])
    zh.extend(['','針對既有恢復版原稿的逐項改稿位置、時間定義修正與英文 Methods/limitations 替換文字，見 [claims_revision_map_zh.md](claims_revision_map_zh.md)。該來源不是已核對的最新投稿原稿，未直接覆寫論文。'])
    zh.extend(['','神經配對差值的跨 seed mean/sample SD 另見 neural_paired_mean_sd.csv。這是三個 optimization seeds 的變異描述，與共用 user/session resamples 的固定模型 bootstrap 區間不同；後者不估計所有可能 optimization seeds 的不確定性。'])
    en.append('Neural paired differences across optimization seeds are additionally summarized by their mean and sample SD in neural_paired_mean_sd.csv. The cluster bootstrap keeps the three trained seed-specific models fixed; it does not estimate uncertainty over the distribution of optimization seeds.')
    (HERE/'report_zh.md').write_text('\n'.join(zh),encoding='utf-8');(HERE/'manuscript_results_en.md').write_text('\n'.join(en),encoding='utf-8')
    (HERE/'results_table.tex').write_text(main.to_latex(index=False,float_format='%.5f'),encoding='utf-8')
    save(HERE/'completion.json',dict(complete=False,reports_generated=True,awaiting_final_validation=True,linear_conditions=12,neural_fits=18,bootstrap_replicates=1500,new_test_datasets=['Coveo','Cosmetics'],config_sha256=sha(HERE/'config.json')))

if __name__=='__main__':main()
