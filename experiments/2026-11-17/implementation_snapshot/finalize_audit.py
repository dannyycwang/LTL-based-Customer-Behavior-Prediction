"""Assemble audit status and a readable report from actual saved outputs."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    for seed in [42,43,44,45]:
        run=ROOT/f'runs/protected_first_v1_seed{seed}_short_raw'
        protected=set(pd.read_csv(run/'protected_formula_pool.csv').session_id_hash)
        sets={}
        for split in ['train','val','test']:
            d=pd.read_csv(run/f'{split}_ids.csv');ids=set(d.session_id);sets[split]=ids
            rows.append(dict(seed=seed,split=split,n=len(d),n_positive=int(d.label.sum()),positive_rate=d.label.mean(),formula_pool_overlap_n=len(ids&protected),duplicated_session_ids=int(d.session_id.duplicated().sum())))
        assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
        assert len(protected)==20000 and protected<=sets['train']
    pd.DataFrame(rows).to_csv(ROOT/'protected_split_audit.csv',index=False)
    c=json.loads((ROOT/'coveo_candidate_diagnostics/metadata.json').read_text())
    m=json.loads((ROOT/'cosmetics_seed42_diagnostics/metadata.json').read_text())
    conflicts=[
      ('C01','Coveo Act+Rel raw inputs','Table4 Short raw uses APC IDs; fusion uses action-only','table4_source_runs.csv; notebook product cell17 stdout; original script planned configs','Confirmed: new five-condition comparison uses APC IDs throughout'),
      ('C02','Coveo formula/test isolation','Reconstructed historical test includes 1808/3019/3006/3013 verified formula-training sessions','coveo_formula_training_overlap.csv; coveo_formula_sample_provenance.json','Confirmed: keep fixed seed42 basis; protect its 20000 fitting/selection sessions in downstream training'),
      ('C03','Formula selection split','Coveo score_formula_with_semantic_key(train_dic,...) contradicts validation-ranking wording','notebook_evidence.csv; product notebook cell10','Keep historical basis; record actual train-based selection honestly; manuscript not edited'),
      ('C04','Coveo first/last raw window','Old product command omits raw_truncation; corresponding local script defaults last','notebook product cell17; product_rawid_vs_ltl_experiment.py:80','Confirmed: new short conditions explicitly first10; original script upload revision still not byte-verified'),
      ('C05','Cosmetics BP/BDP parser','Recorded parser collapses whitespace-event prop trace to one state; saved test Z exactly matches single-state union','cosmetics_parser_real_example.csv; cosmetics_seed42_diagnostics/saved_matrix_scope_check.csv; Dataset3 cell18/19','Confirmed: preserve old results; correct parser and recompute/retrain affected new models'),
      ('C06','Cosmetics checking scope','Stored command explicitly trace_len20/window last, contradicting full-trace description; parser bug prevented intended event window','Dataset3 cell19; notebook_evidence.csv','New Full-LTLf will explicitly evaluate the true full event trace; old result not reusable as true full'),
      ('C07','Purchase observation cutoff','Cosmetics removes purchase events without pre-first-purchase cutoff; Electronics truncates before first purchase','prepare_cosmetics_dec_dataset.py:273; electronics_ltlf_pipeline.py:227','Confirmed: reproduce each original protocol first; separate pre-purchase version later; raw-event checks still incomplete'),
      ('C08','Runtime hardware and scope','Stored Coveo output identifies RTX PRO6000 Blackwell; cost sum excludes formula learning','notebook_evidence.csv; table7_correspondence.csv; product_rawid_vs_ltl_experiment.py:935','Numerical times match; hardware/timing interpretation unresolved in manuscript; no edits made'),
      ('C09','Table5 delta rounding','Coveo Act+Rel delta AUPRC .027803 -> .028 vs paper .027; Act+D+Rel delta F1 .046758 -> .047 vs .046','table5_correspondence.csv','Retain paper values; report exact arithmetic; rounding convention requires confirmation before editing'),
      ('C10','Historical epochs field','Coveo original epochs_ran is assigned best_epoch rather than actual completed epochs','product_rawid_vs_ltl_experiment.py:671; original base script metric update','New runner records actual_epochs_ran separately; old epochs_ran cannot establish actual runtime loop length'),
      ('C11','Electronics Full raw cap','Saved full raw max_len_used is 512, so full need not mean untruncated source session','combined_all_runs (2).csv; supplied Electronics notebook full_max_len512','Preserve explicit cap; determine proportion longer than512 after ready data is obtained')]
    pd.DataFrame(conflicts,columns=['id','topic','evidence_finding','evidence_location','resolution_status']).to_csv(ROOT/'conflicts.csv',index=False)
    statuses=[]
    for ds in ['Coveo','Electronics','Cosmetics']:
        for cond in ['Full raw','Short raw','Short+Prefix-LTLf','Short+Counts','Short+Full-LTLf']:
            statuses.append(dict(dataset=ds,condition=cond,MCC=None,AUPRC=None,status='not trained under confirmed new protocol; do not substitute historical results'))
    pd.DataFrame(statuses).to_csv(ROOT/'new_experiment_status.csv',index=False)
    baseline_path=ROOT/'runs/historical_seed42_short_raw/metrics.json'
    baseline='歷史 baseline 相容性檢查尚在執行；正式新實驗結果尚未產生。'
    if baseline_path.exists():
        b=json.loads(baseline_path.read_text());old=pd.read_csv(ROOT/'table4_source_runs.csv');r=old[(old.dataset=='Coveo')&(old.scheme=='Act+Rel')&(old.condition=='Short raw')&(old.seed==42)].iloc[0]
        comp=[]
        for metric in ['f1','mcc','auprc','brier']:comp.append(dict(metric=metric,historical_saved=float(r[metric]),compatibility_run=float(b[metric]),difference=float(b[metric]-r[metric])))
        pd.DataFrame(comp).to_csv(ROOT/'baseline_compatibility_comparison.csv',index=False)
        baseline=f'歷史 baseline 相容性檢查已完成 {b["actual_epochs_ran"]} epochs；F1={b["f1"]:.6f}、MCC={b["mcc"]:.6f}、AP={b["auprc"]:.6f}、Brier={b["brier"]:.6f}。原腳本在模型建立後才重設 seed，單獨執行的初始化不同，因此不宣稱逐位元重現。差異見 baseline_compatibility_comparison.csv。'
    report=f'''研究程式／論文核對與追加實驗進度（2026-11-10）

本報告依使用者指定的 PDF 草稿、七份 Colab notebook、既有原始程式及保存結果建立。
未修改論文；未覆寫原始資料／notebook／結果；未上傳檔案；未啟動付費雲端。

一、主要交付檔
inventory.csv：各資料集 notebook/scripts/data/splits/formulas/Z/checkpoints/results 的位置與缺項。
notebook_inventory.csv、notebook_evidence.csv：使用者提供的 notebook URL、SHA256、cell 索引及程式／stdout 證據。
table4_source_runs.csv：72 條逐 seed 原結果，附 source 路徑／列號／SHA256。
table4_correspondence.csv：18 組結果全部 mean 與 sample SD（ddof=1）均吻合草稿三位小數。
table5_correspondence.csv：六組增益，保留兩格算術差異；不擅改論文。
table6_correspondence.csv：三組 LTLf-only 均值吻合。
table7_correspondence.csv：四組時間數值吻合一位小數；不表示成本定義或硬體描述正確。
conflicts.csv：逐項衝突、程式位置及確認狀態。
historical_table4.tex：由保存 CSV 自動重建的舊結果表，不能當成新版五條件結果。
new_experiment_status.csv：三資料集五條件共15個正式結果位置；目前未完成，一律留空，沒有假數據。

二、已確認的新版方案
1. Coveo A/B/C/D/E 均使用 action + product ID + category ID 的相同 raw branch。
2. 除 Full raw 外，raw branch 明確使用 first min(10,L)；主設定 K=10、seeds42-45。
3. 保留既有 seed42 basis，不再為每 seed 重學。已用保存 Drive JSON 將20000條公式學習 traces 按順序逐條比對一致。
   這20000個 sessions 亦用於原本公式排名／semantic去重，全部固定在新版 downstream training。
   新版每個 seed train/val/test 仍為179616/38489/38490，正例數和比例與原隨機切分相同。
   protected_split_audit.csv 已核實新 val/test 與公式池交集=0，且三分區 session IDs 不相交。
   無 user IDs，尚不能聲稱 user-level independence。
4. 固定basis是既有 training-ranked basis，不宣稱 validation-ranked 或 prefix-optimized basis。
5. Cosmetics parser 修正已獲確認；舊版另存，修正後重新計算 Prefix/Full 並重訓受影響模型。
6. 各資料集先保留原 observation protocol 以重現，再另建第一次 purchase 前的版本。

三、已實際完成的低成本分析
Coveo 候選CSV原 test 分區：n={c['test_n']:,}，與論文重新隨機切分的38490筆不同，不可混用。
L<=10 全部 Prefix/Full vectors 相同；L>10 n={c['test_long_n']:,}，有{c['long_any_bit_changed_n']:,}筆（{c['long_any_bit_changed_rate']:.2%}）至少一bit改變。
已保存逐公式0->1／1->0率、配對Z、session IDs／labels／lengths及10次shuffle seeds。
shuffle同步移動一個位置的整組propositions，逐條檢查counts與同位置共現保持不變。
位元平行checker經3264次比較與原finite-trace evaluator一致，另檢查strong-next終點。
真實案例按最短L>10改變trace、session ID、第一條改變公式選取，沒有依標籤挑例。
案例的add-to-cart其實在第9步，新增第11步browse使until的左條件在第6步成立；不是新增購物車事件。
詳見 example_witness.json；圖已輸出PDF/320dpi PNG並檢查渲染。

Cosmetics seed42原 test：n={m['test_n']:,}，L>10 n={m['long_n']:,}。
修正解析後，Prefix/true-Full 至少一bit改變的長trace為{m['long_any_changed_n']:,}筆（{m['long_any_changed_rate']:.2%}）。
保存的舊Z與「整段union為單一state」的258700個bits完全一致；與正確full有{m['saved_matrix_vs_true_full_bit_mismatches']:,}個bits不同。
舊notebook雖指定last20，但單一state使事件窗口不生效。不能將保存NPZ稱為正確full或正確last20。
fix_cosmetics_parser.py 已以原notebook parser重現缺陷，並在全部25870筆test驗證修正後事件數等於n_events。
以上是表示診斷，不能解讀為預測效果改善；尚未訓練新Cosmetics predictor。

四、執行與統計
{baseline}
本機GPU為RTX4060 Laptop 8GB。已以真實4096筆training rows跑過原APC模型與Prefix／Counts／Full融合的每種10 batches。
短序列GPU smoke包含synchronization；估時只是小詞彙、純計算估計，不能代替完整執行時間。
Coveo Counts=13個原abstraction propositions + trace length，共14維；LTLf=10維。
原fusion結構保留，Counts入口因此比10維多256個可訓練參數；不宣稱完全相同容量。
Coveo新版使用原evaluator的average_precision_score（AP，非trapezoidal PR-AUC）、Brier/F1/MCC實作。
沿用validation F1 grid選threshold與F1@0.5選checkpoint的原protocol；test labels不參與threshold。
相同threshold用於長度子集；單類子集的MCC/AP/AUC標NA。
訓練entry保存config、ID hashes、basis、vocab、train-only count scaler、epoch logs、last/best checkpoints及逐筆predictions。

五、尚未完成與具體阻擋
1. 三資料集五條件正式predictive comparison、paired prediction differences與完整runtime breakdown尚未完成。
2. Electronics notebook已讀到實際路徑：/content/drive/MyDrive/ltlf_electronics_experiments/electronics_ready_noleak。
   但本機和已連接Drive的folder搜尋未取得這個資料夾，仍需要該資料夾的直接Drive連結／可存取CSV。
   單有notebook的文字路徑不能讀到Colab VM或另一個Drive帳號的檔案。
3. Cosmetics seed42 test、BP clean basis與舊NPZ已下載核對；其餘splits尚未materialize。
   跨seeds重用固定basis前，還需追回formula-training/validation-selection樣本或users，建立不重疊的新評估切分。
4. 三資料集原始purchase截斷、post-purchase事件統計、user overlap與原資料版本尚未全部驗證；不能以候選script代替資料實測。
5. 原Coveo程式epochs_ran填best_epoch；新版actual_epochs_ran才是實際完成迴圈數。
6. Table7的total只含model checking + downstream train/infer，沒有formula learning/selection；不應據此宣稱完整end-to-end成本。

六、重跑入口
先執行 audit_sources.py、audit_secondary_tables.py 重建結果對應；來源路徑在檔案頂部。
audit_coveo_split_overlap.py 需work/coveo_formula_training.json（已讀取的Drive原輸入）驗證公式池與舊切分。
run_coveo.py --dataset coveo --seed 42 --condition short_raw --protocol historical --max-minutes 30
run_coveo.py --dataset coveo --seed 42 --condition prefix_ltlf --protocol protected_first_v1 --prepare-only
condition另支援 full_raw / short_raw / counts / full_ltlf；移除prepare-only才開始training。
既有完成metrics會skip；未完成會載入上一個完整epoch checkpoint續跑；中途超時會保留可續跑狀態。
這一版入口目前支援Coveo，不能冒稱已支援全部資料集；Electronics/Cosmetics需各自已確認的adapter與資料隔離。
diagnose_coveo.py、diagnose_cosmetics.py 由指定資料與basis生成診斷CSV/NPZ。
render_figures.py 只從保存的診斷/結果自動產出圖表，沒有手填結果。
本機scikit-learn等補充依賴安裝在workspace的work/python_packages，沒有改動使用者原Python套件。
'''
    (ROOT/'audit_report.txt').write_text(report,encoding='utf-8')
    print('Finalized audit report; baseline completed:',baseline_path.exists())

if __name__=='__main__':main()
