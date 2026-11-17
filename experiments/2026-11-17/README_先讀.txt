本輪交付：Logic-Guided Temporal Representations for E-commerce Session Prediction
Run ID: next_round_20261116_v1

已完成：36/36 次 fixed-origin 神經訓練（3 datasets × 4 conditions × seeds 42/43/44）；三資料集主要四條件線性模板比較；Coveo/Cosmetics 額外共同排序敏感度；Cosmetics 購買前來源重學習與兩條件線性對照；30 條公式稽核；配對 bootstrap；可編譯修改稿與英文替換文字。
所有正式結果保持原先切分、cohort 與主 frozen basis。沒有依 test 換 seed、threshold、公式或超參數。小型驗證與中止 GPU learner 嘗試保存在 smoke/ 和 cosmetics_source_aborted_gpu*/，不當作正式結果。

主要研究結論：
1. 線性分類器的 LTLf 相對 Counts MCC 增益重現為 +0.0281、+0.0719、+0.0219；三者 paired 95% CI 均高於零。
2. 神經模型沒有呈現對 Counts 的一致新增價值。下列為三個 training seeds 的 mean ± sample SD；不是 bootstrap SE：
   Coveo        raw          MCC 0.4688 ± 0.0031; AP 0.5126 ± 0.0060
   Coveo        counts       MCC 0.4792 ± 0.0061; AP 0.5158 ± 0.0034
   Coveo        transitions  MCC 0.4796 ± 0.0036; AP 0.5125 ± 0.0032
   Coveo        ltlf         MCC 0.4789 ± 0.0024; AP 0.5183 ± 0.0040
   Electronics  raw          MCC 0.2507 ± 0.0289; AP 0.3708 ± 0.0233
   Electronics  counts       MCC 0.2328 ± 0.0354; AP 0.3781 ± 0.0158
   Electronics  transitions  MCC 0.2863 ± 0.0068; AP 0.3599 ± 0.0264
   Electronics  ltlf         MCC 0.1875 ± 0.1285; AP 0.3593 ± 0.0212
   Cosmetics    raw          MCC 0.0356 ± 0.0212; AP 0.3153 ± 0.0088
   Cosmetics    counts       MCC 0.2317 ± 0.0080; AP 0.4360 ± 0.0044
   Cosmetics    transitions  MCC 0.2371 ± 0.0080; AP 0.4377 ± 0.0035
   Cosmetics    ltlf         MCC 0.2312 ± 0.0174; AP 0.4305 ± 0.0026
3. 相對 DECLARE 與 Transitions 的線性 learned-LTLf 差異 CI 都跨零。這不足以支持自動學習優勢，也不是等價證據。
4. Electronics 的 LTLf seed 42 MCC 為 0.0469；原 validation 規則選 threshold 0.65，validation/test 分別預測 14/26 個 positives。重播 checkpoint 與 threshold 均通過，未以較好的 seed 取代此負結果。
5. Cosmetics X(hp) 在 event-10 train/validation 是常數。Electronics 十個 frozen coordinates 在共同過濾下只有四個可用的不同輸出欄。135 對組內公式皆有有限軌跡反例，說明資料上重複不等於邏輯等價。

Cosmetics basis-source 結果：
   frozen_historical: MCC 0.1875; AP 0.4172
   relearned_pre_purchase: MCC 0.1890; AP 0.4142
   pre-purchase minus frozen MCC: +0.0015, paired 95% CI [-0.0096, +0.0132]
   pre-purchase minus frozen AP: -0.0030, paired 95% CI [-0.0063, +0.0005]
   主比較仍使用原 frozen basis；新來源對照不回頭替換主結果。實際 fitting/ranking IDs 透過原 JSON 軌跡順序重播找回，並非使用不相符的舊 sample CSV。購買前無事件的 45/21 個正樣本排除、不補抽，因此非完全相同樣本數的純來源因果對照。
   60 epochs、seed 42、netlen 10、batch 100、lr 0.01、單 restart、300 candidates、eventual-purchase selection target 保留。依訓練資料小測改用 CPU；加總相同損失再 backward 的梯度差為 9.3e-10。最終 cleanup 重播原 notebook：all candidates、移除無實質 atom 公式、依 balanced accuracy/signed gap/positive support 排序、只去相同公式字串，再取前十。不得將檔名 semantic_unique 誤解為邏輯或資料欄位完全去重。

交付位置：
results/experiment_supplement.pdf：英文表格、全部公式與滿足率、語義/來源/計時限制。
results/neural_mean_sample_sd.csv：36 runs 彙整。
results/neural_paired_differences.{csv,pdf,png}：預先固定 seed 42 的配對 95% CI；Coveo session bootstrap，其他 user cluster bootstrap。
results/linear_metrics.csv、linear_paired_differences.*：模板比較及共同排序敏感度。
cosmetics_source/comparison/：basis-source 原始 metrics、逐筆預測、線性模型與 bootstrap。
neural/<dataset>_<condition>_<seed>/：config、best/last checkpoints、test/validation predictions、scaler、vocab、epoch log。
linear/<dataset>/<condition>/：模型、scaler、encoder、逐筆 predictions、config。
prepared/：固定 cohort IDs/labels/users、特徵快取、來源 hash、模板定義、原始與選擇的候選欄位。
audit/：所有公式、原始 ranking rates、first-10 train/validation rates、常數/重複欄、135 個邏輯反例、來源 ID 與計時稽核。
paper_recovered_revision/clean.pdf 與 main.tex：根據找回版本製作的可編譯修改稿。
results/latest_pdf_replacements.tex：以本輪最新 PDF 的具體位置提供可貼入英文/LaTeX 文字。
review_bundle.zip：精簡審閱包；不含約 1 GB checkpoints 或原始資料。完整檔案在上述同層資料夾。

版本與未執行項目：
- 最新附檔 PDF 的 abstract、作者/匿名設定與找回的 LaTeX 不同；未聲稱直接更新最新版 Overleaf。找回稿保持其匿名設定；請以 replacement 文件套用最新版。
- 未找到明確投稿頁數上限。舊稿 9 頁不視為硬性限制；新稿 11 頁，補充材料 7 頁。
- Electronics 共同排序的 learned K=10 額外敏感度無法建立：可用獨特欄位只有四個。主要 frozen 十條公式與四條件比較已完成。
- prefix-trained 公式是可選獨立敏感度，本輪未加入，避免混同來源範圍與 selection target。
- Full raw 的已稽核計時流程本來就使用動態 batch padding；未新增無必要的固定/動態對照。
- Coveo/Electronics 完整歷史 formula-learning + ranking 成本未完全還原；不存在可支持 overall training speedup 的完整流程計時。
- Electronics 原始 split CSV 的逐位元一致性無法證實，且原設定有 train/test user overlap；本輪沿用已稽核重建 split，bootstrap 單位仍為 user。

重現指令（在本資料夾執行；保留原專案、Downloads 模型來源與已稽核資料路徑）：
python prepare.py
python neural.py --smoke --seed 42
python neural.py
python linear.py
python formula_audit.py
python checker_validation.py
python provenance.py
python verify_runs.py
python relearn_cosmetics.py
python source_compare.py
python report.py
python report.py --neural
python update_paper.py
python supplement.py
python write_delivery.py
cd paper_recovered_revision
pdflatex -interaction=nonstopmode -halt-on-error clean.tex
bibtex clean
pdflatex -interaction=nonstopmode -halt-on-error clean.tex
pdflatex -interaction=nonstopmode -halt-on-error clean.tex
cd ..
python finalize.py

neural.py 會跳過完成的 runs，並從最後完整 epoch（含 optimizer/RNG）續跑未完成 run。relearn_cosmetics.py 若沒有完成 basis，會固定 seed 從頭重學，不把部分訓練模型冒充完成模型。原始專案與既有實驗未覆寫。
語義參考：https://www.cs.rice.edu/~vardi/papers/ijcai13.pdf；DECLARE encoding：https://ceur-ws.org/Vol-3310/paper9.pdf。新 baseline 是四類 Boolean 模板與固定 K/support-gap 選擇，不是原文 Fisher/coverage、可區分 vacuity 的整套實作。