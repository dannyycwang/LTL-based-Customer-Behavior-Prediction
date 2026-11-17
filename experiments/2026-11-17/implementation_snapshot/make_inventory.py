"""Inventory/evidence receipts from inspected local files and supplied notebooks."""
from pathlib import Path
import json,hashlib
import pandas as pd
from diagnose_coveo import vector,freeze
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]/'work';H=Path('C:/Users/ycw')

def main():
    specs=[('colab_0','CoveoAblation','1R8nds9m8nGFFuXl-JyO_5l1YHdLn3Gx2'),('colab_1','LTLexperiment_product','1H7886vVWP4mV0KAxmFUDdhgRe6Xp43aQ'),('colab_2','LTL_Experiment_behavior_Dwell','18EvuBkfj9bQViAWtZ2EoQ9LktdG9-WEp'),('colab_3','LTLrulesExtract_browseonly','1cL3ChAwFIGQZ3d-gwWBEUMR00C4GWs0d'),('colab_4','LTL_experiment','1cut0es6RnwjTWQps4nSyJcJSp47Q0DHB'),('electronics_nb_0','Dataset3 (Cosmetics)','1YT6kOoGS1XXh2JRz6rDIOvVkg2B521rI'),('electronics_nb_1','LTL_dataset2 (Electronics)','1jH8c9EYHgKJoT4m3GM-SDQQ-tFnJjjr2')]
    rows=[];evidence=[]
    terms=['score_formula_with_semantic_key(train_dic','make_stratified_splits','--raw_truncation','--trace_len 20','--window last','trace_len: 20','window: last','NVIDIA RTX PRO 6000','pc_no_dwell_fixedF42_fullcheck_top10','electronics_ready_noleak','full_max_len','--threshold_mode','epochs_ran','return [split_state_string(s)]']
    for stem,title,id in specs:
        f=WORK/(stem+'.ipynb');obj=json.loads(f.read_text(encoding='utf-8'))
        rows.append(dict(title=title,url='https://colab.research.google.com/drive/'+id,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),n_cells=len(obj['cells']),n_code_cells=sum(c['cell_type']=='code' for c in obj['cells'])))
        for i,c in enumerate(obj['cells']):
            for where,txt in [('source',''.join(c.get('source',[]))),('stored_stdout',''.join(''.join(o.get('text',[])) for o in c.get('outputs',[])))]:
                lines=txt.splitlines()
                for j,line in enumerate(lines):
                    if any(t in line for t in terms):evidence.append(dict(notebook=title,url='https://colab.research.google.com/drive/'+id,cell_index_zero_based=i,part=where,line_in_cell_1based=j+1,excerpt='\n'.join(lines[max(0,j-2):min(len(lines),j+3)])))
    pd.DataFrame(rows).to_csv(ROOT/'notebook_inventory.csv',index=False)
    pd.DataFrame(evidence).to_csv(ROOT/'notebook_evidence.csv',index=False)
    items=[
        ('Coveo','processed data','C:/Users/ycw/Downloads/browsing_draw_random4_sessions_product_compact.csv','found','256595 sessions; original split column differs from downstream aligned random splits'),
        ('Coveo','raw event candidates','C:/Users/ycw/Downloads/browsing_train.csv','candidate','Raw log version/row alignment with processed sessions not fully verified'),
        ('Coveo','splits','reconstructed_split_seed_{42,43,44,45}.npz','reconstructed','70/15/15 notebook recipe; exact stored original ID files not acquired'),
        ('Coveo','formula basis','C:/Users/ycw/Downloads/Z_APC_top20_formulas.json','verified','Byte-identical to Drive seed-42/window_last30 top20 semantic-unique basis; K10 uses first ten'),
        ('Coveo','formula learning input','https://drive.google.com/file/d/1U-qME5NKoPbMNGr4-Svf67qOjJHRNyRS/view','verified','20000 labeled traces matched local reconstruction in order'),
        ('Coveo','checkpoints','runs/historical_seed42_short_raw','new local','Original downstream checkpoints not located; compatibility run saves new checkpoints'),
        ('Electronics','scripts','C:/Users/ycw/Downloads/electronics_ltlf_pipeline.py','found','Also first-window runner and dwell-leakage hotfix; notebook shows noleak runs'),
        ('Electronics','processed data/splits','/content/drive/MyDrive/ltlf_electronics_experiments/electronics_ready_noleak','missing locally and connector folder search empty','Path confirmed from supplied LTL_dataset2 notebook; actual accessible file/folder URL still required'),
        ('Electronics','formula basis','C:/Users/ycw/OneDrive - Høgskulen på Vestlandet/Desktop/LTL experiment/dataset2/BP_semantic_unique_top10_formulas.json','found','Fixed seed42 basis; formula-fitting and selection IDs must be recovered before cross-seed reuse'),
        ('Electronics','results','C:/Users/ycw/OneDrive - Høgskulen på Vestlandet/Desktop/LTL experiment/dataset2/combined_all_runs (2).csv','verified arithmetic','Table4-6 means match; full raw saved max_len_used=512; original IDs/predictions/checkpoints unavailable'),
        ('Cosmetics','scripts','C:/Users/ycw/Downloads/train_cosmetics_first_window_clean_names_v6_withBD_len20_fullfusion.py','found','Source raw pack mapping confirmed; stored result thresholds fixed 0.5'),
        ('Cosmetics','data/splits','https://drive.google.com/drive/folders/1VP0rNpV1ZnAgSzObn6Ydvt6VAr864-Sm','found in connected Drive','seed42 test downloaded and inspected; train/val and seeds43-45 not materialized'),
        ('Cosmetics','formula basis','https://drive.google.com/file/d/1awcLK_t5i7cZ9uqfa_h7zVd-tF_erGIq/view','found and downloaded','BP clean top20; first10 formula strings match saved NPZ'),
        ('Cosmetics','satisfaction','https://drive.google.com/file/d/1ymyncE1tTWxDskVtjvk4RMIbS7bdjrqL/view','verified parser defect','258700/258700 test bits match single-state union; true full differs in 29613 bits'),
        ('Cosmetics','raw logs','2019-Dec.csv from Kaggle as recorded in Dataset3 notebook','not materialized','Need raw purchase event positions to quantify retained post-purchase events or rebuild pre-purchase version'),
        ('all','new predictions/results','runs/*/predictions.csv','only historical Coveo seed42 authorized for full training','No new five-condition results for three datasets yet; missing cells must remain NA')]
    pd.DataFrame(items,columns=['dataset','item','location','status','notes']).to_csv(ROOT/'inventory.csv',index=False)
    e=json.loads((ROOT/'coveo_candidate_diagnostics/example.json').read_text());tree=freeze(e['formula']['formula_tree']);tr=[set(s) for s in e['trace']]
    witnesses=[]
    assert tree[0]=='U'
    for t in range(len(tr)):
        if vector(tr[t:],[tree[2]])[0] and all(vector(tr[k:],[tree[1]])[0] for k in range(t)):
            witnesses.append(t+1)
    explanation=dict(session_id=e['session_id'],formula=e['formula']['formula_str'],full_outer_until_witness_positions_1based=witnesses,prefix_value=int(vector(tr[:10],[tree])[0]),full_value=int(vector(tr,[tree])[0]),left_subformula_at_position6_prefix=int(vector(tr[5:10],[tree[1]])[0]),left_subformula_at_position6_full=int(vector(tr[5:],[tree[1]])[0]),explanation='Cart action a is already at position 9. At outer-until witness position 7, X(X(a)) is true. The left condition must hold at positions 1-6; at position 6, F(X^5(b)) requires a b at position >=11. The b at position 11 makes this true in the full trace; the prefix ends at 10. This example demonstrates an endpoint/continuation effect, not a newly observed cart action.')
    (ROOT/'coveo_candidate_diagnostics/example_witness.json').write_text(json.dumps(explanation,indent=2),encoding='utf-8')
    print('Inventory, notebook evidence and checked example witness written.')

if __name__=='__main__':main()
