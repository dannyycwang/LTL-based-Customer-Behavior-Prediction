"""Create text/CSV/LaTeX results without modifying the manuscript."""
import json
import ast
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
RUNS={'Coveo':'coveo_v2','Electronics':'electronics_v2','Cosmetics':'cosmetics_v3'}

def main():
    scores=[];uncertainty=[];cost=[];formulas=[];diagnostics=[];mapping=[]
    for dataset,run in RUNS.items():
        root=HERE/'results'/run
        provenance=json.loads((root/'provenance.json').read_text())
        lookup={json.dumps(r['formula_tree']):j for j,r in enumerate(provenance['candidate_bank'])}
        for i,r in pd.read_csv(provenance['candidate_file']).iterrows():
            tree=ast.literal_eval(r.formula_tree)
            mapping.append(dict(dataset=dataset,export_row=i,canonical_candidate=lookup[json.dumps(tree)],
                                formula=r.formula_str))
        for protocol in ['full_observation','next_10']:
            p=root/protocol
            for filename,collector in [('metrics.csv',scores),('bootstrap.csv',uncertainty),
                                       ('selected_formulas.csv',formulas),('candidate_diagnostics.csv',diagnostics)]:
                f=pd.read_csv(p/filename)
                f['dataset']=dataset;f['protocol']=protocol
                collector.append(f)
            s=json.loads((p/'selection.json').read_text())
            cost.append(dict(dataset=dataset,protocol=protocol,selection_seconds=s['seconds'],
                             eligible_candidates=s['eligible_count'],
                             original_top10_retained=len(set(s['indices'])&set(provenance['original_indices'])),
                             validation_log_loss_gain=s['baseline_validation_log_loss']-s['final_validation_log_loss']))
    scores=pd.concat(scores,ignore_index=True);u=pd.concat(uncertainty,ignore_index=True)
    scores.to_csv(HERE/'results_metrics.csv',index=False)
    u.to_csv(HERE/'results_uncertainty.csv',index=False)
    pd.DataFrame(cost).to_csv(HERE/'selection_overhead.csv',index=False)
    pd.concat(formulas,ignore_index=True).to_csv(HERE/'selected_formula_comparison.csv',index=False)
    pd.concat(diagnostics,ignore_index=True).to_csv(HERE/'all_candidate_diagnostics.csv',index=False)
    pd.DataFrame(mapping).to_csv(HERE/'candidate_row_mapping.csv',index=False)
    lines=['Residual-guided LTLf selection: research report',
           'Completed matched evaluation; existing manuscript and results unchanged.', '',
           'A = raw prefix + Counts + Transitions; B = A + original Top-10;',
           'C = A + residual Top-10. Electronics D = A + retrained standalone Top-10.',
           'All entries use the same audited split, classifier and observation boundary.', '',
           'RESULTS (estimate +/- bootstrap standard error; not a 95% interval)']
    table=[]
    for dataset in RUNS:
        for protocol in ['full_observation','next_10']:
            group=u[(u.dataset==dataset)&(u.protocol==protocol)]
            for condition in ['A','B','C']+(['D'] if dataset=='Electronics' else []):
                point=group[(group['first']==condition)&group.second.isna()].set_index('metric')
                mcc=point.loc['MCC'];ap=point.loc['AP']
                table.append(dict(dataset=dataset,protocol=protocol,condition=condition,
                    MCC=f"{mcc.estimate:.4f} +/- {mcc.se:.4f}",AP=f"{ap.estimate:.4f} +/- {ap.se:.4f}"))
    tab=pd.DataFrame(table);lines.append(tab.to_string(index=False));lines+=['','PAIRED DIFFERENCES (95% percentile intervals)']
    differences=u[u.second.notna()]
    lines.append(differences[['dataset','protocol','first','second','metric','estimate','lower95','upper95']].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    tab.to_csv(HERE/'publication_table.csv',index=False)
    (HERE/'publication_table.tex').write_text(tab.to_latex(index=False,escape=True),encoding='utf-8')
    lines+=['','CROSS-DATASET ASSESSMENT']
    for second in ['A','B']:
        for metric in ['MCC','AP']:
            d=differences[(differences['first']=='C')&(differences.second==second)&(differences.metric==metric)]
            lines.append(f'C minus {second}, {metric}: {int((d.estimate>0).sum())}/{len(d)} positive point estimates; '
                         f'{int((d.lower95>0).sum())} intervals wholly positive; {int((d.upper95<0).sum())} wholly negative.')
    lines+=['','SELECTION COST',pd.DataFrame(cost).to_string(index=False),
            '', 'METHOD AND INTERPRETATION',
            'Selection maximizes validation log-loss reduction with a C=1 logistic model',
            'conditional on Counts + Transitions. Raw-prefix features are included in all',
            'final classifiers, not in the selection proxy. K=10 is fixed, even if later',
            'development gains are negative. Exact training duplicates and constants are',
            'excluded; each exclusion and every evaluated step are saved.',
            'The original marginal class-separation ranking can favor count/length proxies.',
            'Per-candidate diagnostics quantify conditional gains instead of assuming that',
            'additional temporal operators necessarily supply additional predictive value.',
            'Training-only equal-summary witnesses and correlation files are provided in',
            'each dataset/protocol directory. Redundancy diagnostics compare both the',
            'selected ten and ALL adjacent proposition-pair occurrence bits. Saved witness',
            'pairs match counts and ALL such bits. This does not establish information',
            'absent from raw events or transition-frequency features. A length predicate or thresholded',
            'count can aid a linear classifier without adding temporal information.',
            '', 'ELECTRONICS RETRAINING',
            'The missing candidate pool was regenerated from the historical training recipe:',
            '8000 training traces, seed 42, last 10 events, BP/Act+Rel, 50 epochs.',
            'Training took 601.95 seconds and extraction 34.21 seconds on the available GPU.',
            '250 candidates, 122 fitting-sample signatures; 49344 independent semantic',
            'checks passed and all 250 support pairs were reproduced. This is a new model,',
            'not an exact restoration of the missing historical checkpoint. However, its',
            'standalone Top-10 has exactly the same ordered formula structures as the',
            'historical basis: B and D predictions are identical in both protocols.',
            '', 'SELECTED FORMULA INTERPRETATIONS',
            'Electronics full: X(F d) requires a detail event after the first event;',
            'it is not just any-detail presence. !(F X X a) excludes cart events at',
            'positions three and later. X X true is merely a length >=3 indicator:',
            'it can help a linear classifier but introduces no information beyond length.',
            'Cosmetics full: d U r requires a removal witness preceded only by detail',
            'events from the start. A U B is strong until: B must occur, including the',
            'possibility of an immediate witness. Nested expressions must not be casually',
            'described as strict sequences or long-distance dependencies without checking',
            'these boundary semantics. The full selected lists are in the comparison CSV.',
            'A verified Electronics training pair has action sequences D D A D D and',
            'D D D A D, with identical Act+Rel counts and ALL adjacent proposition-pair',
            'occurrence bits after excluding dwell atoms. X(pc U X(a)) distinguishes',
            'them. In this pair pc is false at the second position, so the distinction',
            'is whether the third event is cart. This is concrete positional information',
            'beyond those summaries, not evidence that the benefit requires a long gap.',
            '', 'ASSESSMENT OF THE PAPER CONTRIBUTION',
            'Cosmetics full observation improves over the original basis in both MCC and',
            'AP with paired intervals above zero. Electronics full improves AP, while',
            'its MCC difference from the original basis remains uncertain. Coveo also',
            'improves full-observation AP (delta 0.00340, 95% interval 0.00136 to',
            '0.00566), but its MCC difference is uncertain (delta 0.00062, interval',
            '-0.00041 to 0.00171). All three full-observation AP differences are positive',
            'with marginal paired intervals excluding zero. Only Cosmetics has a clear',
            'MCC improvement over the original basis in this experiment. Event-10',
            'results for these datasets do not establish consistent improvement.',
            'The method therefore supports a qualified feature-selection ablation rather',
            'than a general superiority claim over behavioral summaries. In particular,',
            'Cosmetics full observation has no selected-formula witness in the observed',
            'training data when ALL adjacent occurrence bits and counts are held equal.',
            'Its improvement may reflect omitted adjacent features or nonlinear summaries',
            'better suited to the linear predictor, rather than new nonlocal information.',
            'Electronics has such all-adjacent witnesses for 2/10 full and 4/10 event-10',
            'selected formulas; Coveo has 7/10 and 8/10, respectively. Cosmetics event-10',
            'has 2/10. An absence of observed witnesses is not proof of logical equivalence.',
            'This is finite-sample representation evidence, not proof',
            'of predictive usefulness of those individual nonlocal distinctions.',
            'Selection has measurable one-time cost and no increase in feature count at',
            'deployment. Complexity is justified only for settings with reliable benefit;',
            'the current event-10 evidence does not justify an accuracy-based blanket claim.',
            '', 'LIMITATIONS',
            'One split, one formula-training seed, one fixed classifier; no claim of broad',
            'model or split robustness. Bootstrap holds models, thresholds and selected',
            'formulas fixed: it quantifies test sampling, not selection/mining uncertainty.',
            'Electronics and Cosmetics resample users; Coveo resamples sessions. Each',
            'comparison shares bootstrap draws. Intervals are marginal, not adjusted for',
            'the multiple datasets, metrics and comparisons.',
            'Fixed-origin selection uses its future-purchase development target whereas',
            'the historical basis was frozen for session purchase. Thus this contrast also',
            'includes target adaptation. Cosmetics historical formula fitting/ranking had',
            'some purchase-masked histories extending past purchase; preserve and disclose',
            'that historical transfer limitation rather than silently changing the cohort.',
            'The selection proxy omits raw prefix features that enter the final classifier.',
            'A formula valuable beyond Counts+Transitions can be redundant with that raw',
            'prefix. Log loss also need not optimize MCC or AP. Small event-10 development',
            'cohorts, greedy multiple comparisons and the marginally mined candidate pools',
            'are plausible reasons for uncertain or negative early-prediction results.',
            'Full-observation cutoffs are purchase-relative, unlike event-10 prediction;',
            'length and boundary features can exploit this protocol without demonstrating',
            'general early-prediction value. No result-driven protocol changes were made.',
            'Electronics shares users across splits; user-bootstrap uncertainty does not',
            'turn its session-generalization evaluation into an unseen-user evaluation.',
            'The test cohorts have been used in the earlier paper development. A new cohort',
            'would be required for stronger independent confirmation.',
            'Selection runtimes are measured wall time with concurrent workloads; they are',
            'not isolated hardware benchmarks. Cached preparation and checker validation',
            'must not be compared as if they measured identical feature-construction work.',
            'Small effects or intervals spanning zero do not establish an improvement.',
            'A/B replay matches sample IDs and labels but is not bit-identical to historical',
            'probabilities. Maximum absolute replay differences are saved per run; all',
            'contrasts here use newly fitted matched A/B/C models, never mixed old/new fits.',
            'Results support a selection-method claim only to the extent that conditional',
            'held-out gains are consistent and scientifically meaningful. Inspect the',
            'paired intervals above before trading substantial selection cost for benefit.',
            '', 'REPRODUCIBILITY',
            'See README.txt, retrain_electronics.py, run_residual.py, manifests, selection',
            'traces, cached features, predictions, historical_replay.csv and bootstrap.csv.',
            'Interrupted v1/v2 development runs are excluded from this report.']
    (HERE/'research_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(tab.to_string(index=False))

if __name__=='__main__':main()
