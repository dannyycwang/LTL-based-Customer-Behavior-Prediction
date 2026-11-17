"""Minimal revision of recovered source; latest uploaded PDF is separately mapped."""
from common import *
import shutil,re,difflib

def main():
    original=PROJECT/'work/paper_thematic/source';dest=HERE/'paper_recovered_revision'
    if not dest.exists():
        shutil.copytree(original,dest,ignore=shutil.ignore_patterns('*.pdf','*.png','*.svg','*.aux','*.log','*.out','*.bbl','evidence','statistics','scripts','template_source'))
        # Copy only assets actually used by this recovered manuscript.
        for p in (original/'figures').glob('*.pdf'):shutil.copy2(p,dest/'figures'/p.name)
        for p in original.glob('*.bbl'):shutil.copy2(p,dest/p.name)
    scores=pd.read_csv(HERE/'results/neural_mean_sample_sd.csv').set_index(['dataset','condition'])
    table=[r'\begin{tabular}{llcc}',r'\toprule Dataset & Input & MCC & AP \\',r'\midrule']
    names={'raw':'Raw','counts':'+ Counts','transitions':'+ Counts + Transitions','ltlf':'+ Counts + LTLf'}
    for ds in ['Coveo','Electronics','Cosmetics']:
        for c in names:
            r=scores.loc[(ds,c)];table.append(f"{ds if c=='raw' else ''} & {names[c]} & ${r.MCC_mean:.3f}\\pm{r.MCC_std:.3f}$ & ${r.AP_mean:.3f}\\pm{r.AP_std:.3f}$ \\\\")
        table.append(r'\midrule')
    table[-1]=r'\bottomrule';table.append(r'\end{tabular}')
    (dest/'tables/fixed_neural.tex').write_text('\n'.join(table),encoding='utf-8')
    shutil.copy2(HERE/'results/neural_paired_differences.pdf',dest/'figures/fixed_neural_differences.pdf')
    text=(original/'main.tex').read_text(encoding='utf-8')
    abstract=r'''\rev{We study clickstream purchase prediction using action and product/category-relation traces (Act+Rel). A frozen basis of learned linear temporal logic over finite traces ($\mathrm{LTL}_f$) formulas provides explicit Boolean summaries alongside a short raw sequence. Across three e-commerce datasets, extending formula checking beyond a short raw window recovers information from longer histories. At a common event-10 origin, logic improves count-augmented logistic regression, but a new 36-run neural comparison does not show a consistent incremental benefit over Counts. Dimension-matched adjacent transitions and a small DECLARE-template baseline are competitive; the results do not establish an advantage from automatic formula learning. Several learned coordinates encode occurrence or simple guarded temporal conditions, and some are redundant on observed prefixes. Short raw inputs reduce measured neural-update cost and GPU allocation, while feature construction can increase inference latency. These findings delimit the predictive and computational value of explicit temporal summaries without claiming an overall training speedup.}'''
    abstract=abstract.replace('using action and product/category-relation traces','using traces of actions and product and category relations')
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',text,flags=re.S)
    # Replace the old horizon plot in the main text, preserving its underlying source in the package.
    pattern=r'\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[.*?\]\{figures/rq3_horizon.pdf\}.*?\\end\{figure\*\}'
    replacement=r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fixed_neural_differences.pdf}
\caption{\rev{Incremental neural value of LTLf at event 10 with $H=10$. Points use the prespecified seed-42 models and validation-selected thresholds; bars are paired percentile 95\% confidence intervals from 1,500 resamples. Coveo resamples sessions; Electronics and Cosmetics resample users. These conditional intervals are distinct from the training-seed SD in Table~\ref{tab:fixed-neural-new}.}}
\Description{Paired MCC and AP differences from Counts and Transitions, including negative Electronics MCC differences.}
\label{fig:fixed-neural-new}
\end{figure*}'''
    matches=list(re.finditer(pattern,text,flags=re.S))
    # Anchor at the nearest figure start so earlier figures cannot be consumed.
    target=text.index('{figures/rq3_horizon.pdf}');start=text.rfind('\\begin{figure*}',0,target);end=text.index('\\end{figure*}',target)+len('\\end{figure*}')
    text=text[:start]+replacement+text[end:]
    text=text.replace('(Figure~\\ref{fig:horizons})','(retained in the supplementary horizon results)')
    addition=r'''\subsection{Matched Neural and Template Controls}
\begin{table}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{\rev{Fixed-origin neural prediction, event 10 and $H=10$. All rows include the same first-ten-event raw encoder within each dataset. Values are mean $\pm$ sample SD over training seeds 42, 43, and 44.}}
\label{tab:fixed-neural-new}
\resizebox{\columnwidth}{!}{\input{tables/fixed_neural}}
\end{table}
The new comparison holds the observation boundary, raw encoder, 64-wide classification head, 25-epoch cap, patience 5, batch size 256, and optimizer settings fixed across four conditions. Vocabularies and scalers use training prefixes only. Validation AP selects checkpoints, followed by a validation-MCC threshold grid. These rules define the new controlled comparison and differ from some historical full-observation settings. The split and frozen formula bases remain unchanged; each summary uses only events 1--10.

Table~\ref{tab:fixed-neural-new} does not show a stable incremental benefit from LTLf over Counts. The mean MCC differences are $-0.0003$, $-0.0453$, and $-0.0005$ for Coveo, Electronics, and Cosmetics. Electronics has substantial initialization variation: LTLf MCC is $0.187\pm0.128$. Its prespecified seed-42 MCC difference from Counts is $-0.206$ (paired 95\% CI $[-0.304,-0.108]$), whereas its AP difference is $-0.015$ ($[-0.039,0.003]$). Figure~\ref{fig:fixed-neural-new} reports all primary paired contrasts. This fixed-model result is not a statement about the mean across initializations, and intervals crossing zero are not evidence of equivalence.

Under the Table~\ref{tab:linear} logistic setup, ten selected Boolean DECLARE-style coordinates yield MCC $0.470$, $0.266$, and $0.201$, compared with frozen learned LTLf $0.475$, $0.289$, and $0.188$. Existence, response, strict precedence, and chain-response candidates use the same proposition vocabulary. Template and re-ranked controls share training ranking IDs, eventual-purchase labels, an absolute-support-gap objective, and empirical-column deduplication. The frozen basis retains its historical selection protocol, so candidate generation is not fully matched. All learned-versus-template MCC and AP intervals cross zero; these experiments do not establish added value from automatic learning. The Electronics pool has only four usable distinct columns after the common prefix filter, precluding its optional ten-feature re-ranking sensitivity arm.

\begin{table}[t]
\centering\footnotesize
\caption{\rev{Representative actual frozen formulas; $a$ is add-to-cart. Full original syntax, provenance, and train/validation satisfaction rates appear in the experiment supplement.}}
\label{tab:actual-formulas}
\begin{tabular}{lll}\toprule Dataset/index & Finite-trace form & Interpretation \\
\midrule
Coveo/3 & $\mathbf{F}\mathbf{X}^{2}a$ & Cart at event 3 or later \\
Electronics/1 & $\mathbf{X}\mathbf{F}a$ & Cart after event 1 \\
Cosmetics/1 & $hp\,\mathbf{U}\,a$ & Product-present until cart \\
Cosmetics/6 & $\mathbf{X}hp$ & Product at event 2 \\
\bottomrule\end{tabular}
\end{table}
Exact finite-trace checking distinguishes all pairs within each ten-formula basis, yet observed columns can coincide. Cosmetics $\mathbf{X}hp$ is constant in the event-10 train/validation cohort, and $hp\,\mathbf{U}\,a$ coincides there with cart occurrence. Such empirical identities are not logical equivalences over arbitrary multi-proposition traces. These formulas support explicit occurrence and guarded temporal summaries, not a claim that complex long-range dependencies were learned.

'''
    # Check recovered table label before insertion.
    linear_labels=re.findall(r'\\label\{([^}]*linear[^}]*)\}',text)
    if linear_labels:addition=addition.replace('tab:linear',linear_labels[0])
    addition += r'''Prior DECLARE encoding~\cite{di2022making} separates violations from satisfaction and can distinguish vacuous cases, and combines Fisher-score ranking with a coverage rule. Our small baseline instead merges both forms of satisfaction into one Boolean value and selects ten columns by the declared support-gap rule. The experimental distinction is therefore the candidate basis and its integration with a short raw encoder, not the invention of logic-based trace encoding.

'''
    addition += r"""A separate Cosmetics source control relearns formulas from pre-purchase histories while retaining eventual-purchase targets and the recovered learner settings. The frozen and relearned bases yield MCC 0.1875 and 0.1890, and AP 0.4172 and 0.4142. The user-cluster paired differences are $+0.0015$ (95\% CI $[-0.0096,0.0132]$) for MCC and $-0.0030$ ($[-0.0063,0.0005]$) for AP. Neither establishes a source benefit. Empty pre-purchase traces exclude 45 fitting and 21 ranking sessions without replacement; CPU execution also differs from the historical GPU run. This is a separate refit, not a replacement selected by test scores. Details and the full basis-source table appear in the supplement.

"""
    text=text.replace('\\subsection{Computational Cost and Efficiency}',addition+'\\subsection{Computational Cost and Efficiency}')
    text=text.replace('At a common event boundary, logic adds useful information to counts, with adjacent transitions providing a competitive compact representation.','At a common event boundary, logic adds information to count-augmented linear classifiers, while the matched neural comparison shows no consistent incremental gain. Adjacent transitions and fixed templates remain competitive, and an automatic-learning advantage is not established.')
    text=text.replace('temporal summaries complement counts at a shared boundary','temporal summaries complement counts in linear models at a shared boundary')
    cost=(original/'tables/cost.tex').read_text(encoding='utf-8').replace('@{}llrrrr@{}','@{}llr@{\\hspace{14pt}}rrr@{}')
    (dest/'tables/cost.tex').write_text(cost,encoding='utf-8')
    text=text.replace('\\balance', '')
    (dest/'main.tex').write_text(text,encoding='utf-8');(dest/'clean.tex').write_text('\\def\\MakeCleanCopy{1}\n\\input{main.tex}\n',encoding='utf-8')
    (HERE/'results/recovered_source.diff').write_text(''.join(difflib.unified_diff((original/'main.tex').read_text(encoding='utf-8').splitlines(True),text.splitlines(True),fromfile='recovered/main.tex',tofile='revision/main.tex')),encoding='utf-8')
    (HERE/'results/latest_pdf_replacements.tex').write_text('% Latest uploaded PDF: replace Abstract (page 1, lines 16-29).\n'+abstract+'\n\n% Insert after Section 4.3 and replace the old horizon figure with the new paired-difference plot.\n'+addition+'\n% Replace the common-boundary claim in Conclusion (pages 8-9):\nAt a common event boundary, logic improves count-augmented linear classifiers, but the matched neural comparison provides no consistent incremental benefit. Transitions and fixed DECLARE-style templates remain competitive; automatic formula learning has no established advantage in these experiments.\n',encoding='utf-8')
    save_json(HERE/'results/manuscript_version_note.json',dict(latest_pdf='C:/Users/ycw/Downloads/Logic_Guided_Temporal_Representations_for_E_commerce_Session_Prediction__2_.pdf',recovered_source=str(original/'main.tex'),exact_latest_source=False,differences='Latest PDF abstract and author/anonymity settings differ from recovered source. Revision is explicitly based on recovered source, not a claim of updating the latest Overleaf version.',page_limit='No explicit venue page limit recovered; prior package has 9 pages, which is not treated as a submission limit.'))
if __name__=='__main__':main()
