# 執行前／執行中協議澄清

數值／執行效能檢查：`solver_check.json` 只比較同一 L2 logistic objective 的 liblinear primal/dual 求解，不計算 AP/MCC、不讀取 test。Dual 在 2,000 iterations 未收斂，且不符合事前數值一致性與兩倍加速門檻，因此沒有採用；正式 A/B/C/D 全數仍使用既有 primal liblinear。這不是增加一個 classifier 條件。

GPU 的同 objective 軟體檢查也未採用：GPU 實測 96.36 秒、CPU 58.05 秒，且 GPU 到達 2,000 iteration 上限，未達事前門檻。Objective 與預測接近並不構成加速證據。`gpu_solver_check.json` 保存完整實測結果；沒有使用 GPU check 的模型、公式選擇或預測。兩項軟體檢查均不改正式 LR 設定，沒有 test AP/MCC 搜尋。正式 selector 從已保存步驟原樣續跑。

所有統計任務參數（t、m、H、seed、logistic objective、metrics、candidate budgets、learner epochs、selection loss/budget/stopping）均未因效果改動。求解器實作與數值精度有以下明列差異。

1. 初始 config 的 Cosmetics boundary policy 暫記 `PENDING_USER_LABEL_CLARIFICATION`。已完成原始 December replay，但尚未用暫定的 24 小時方案產生 cohort、mining 或結果。使用者提供原 Kaggle 來源後，確認可取得全部五個月份；採用可驗證的跨月 session 排除，寫入最終 config，再建立 Cosmetics cohort 與開始唯一 mining run。Coveo 的先行 mining manifest 仍保留其啟動當時 config SHA；此澄清完全不更動 Coveo 的資料或 learner。
2. 初始 selection 使用兩個候選 fit；底層數值 thread pool 固定一個。Cosmetics t=10 B 的前兩步保存後，加入原樣續跑及 baseline/winning-model 重用，改為每次只執行一個 dataset selector、最多四個並行候選。已完成的選擇保持原樣；續跑會重新擬合並驗證 baseline 與已選模型的 validation loss 與紀錄相符至 1e−12。這只改 16GB 本機的工作排程與避免重複擬合，不改候選集合、C、solver、threshold 或選擇準則；未依 test performance 調整。
3. 繼承的模型、feature、source hash checks 保留。新增摘要與窗口抽象通過全部實際 prefix replay；未修改原資料、模型、報告、論文或 remote。
4. 執行期間重跑的 synthetic validator 不是研究實驗；不使用真實 labels 選擇方法，也不產生新增 mining seed。
5. 第一個 Cosmetics mining attempt 在 epoch 10 的 progress print 遇到 Windows CP950 無法編碼路徑中的挪威文字元而終止；當時 Coveo 尚在執行，也會遇到同一輸出問題，因此一併停止。未完成的 checkpoints、輸入與 console logs 保存於各 dataset 的 `mining_failed_console_encoding/`，沒有 extraction、selection 或 test results。修正 stdout 為 UTF-8 後以相同 seed、sample IDs、training JSON、learner 與 epochs 從頭重跑；這是技術失敗重跑，不是新增候選池或根據效果調整方法。唯一進入比較的 pool 為後續成功完成的版本。

原有 split 建立在歷史抽樣後的 session cohort；本輪沿用而不重抽。AP 與 Brier 描述這些 cohorts，不能當成自然部署流量下的 prevalence 或 calibration 估計。

6. 舊 sklearn 冷啟動在 Cosmetics t10 每候選約一分鐘，DECLARE 過濾後仍有 378 個候選。先檢查 dual 與 GPU 同 objective 實作，分別未收斂／未通過速度門檻，均未採用。官方 LIBLINEAR 2.50.0 支援同 primal L2 logistic objective 的初始解；eps=1e−4 首次檢查未通過預設預測差門檻，保留於 `native_check_eps1e4.json`。提高求解精度至 eps=1e−6 後，在相同 A 與預先指定第一個 eligible candidate 上通過原門檻：relative objective difference<1e−6、validation max prediction difference<.002、mean<1e−4、解析 gradient 收斂；候選擬合快約 5.5 倍。沒有 AP/MCC/test 比較。四條件正式流程全部改用該版本；舊 Cosmetics t10 部分選擇封存在 `execution_archive/sklearn_partial/t10`，同 pool/cohort 從頭選擇，避免混用數值結果。此差異可能影響微小 validation gain 的排序，結論限定本次固定實作。選擇 checkpoint 保存實際 fitted model，續跑直接重用並核對 validation loss，而非用不同初始解重新求解。
7. Windows 記錄確認 2026-11-19 15:22:52 UTC 至 2026-11-20 09:36:03 UTC 睡眠，見 `system_sleep.json`。跨越該區間的原始 wall time 包含睡眠，不能當作 active compute time；不宣稱方法速度或記憶體優勢。

即使 C 超過 A，本輪只證明固定 logistic regression 下的額外預測表現；新增 Boolean transformations 也可能幫助線性模型利用既有摘要隱含的資訊，不能僅由 performance 宣稱 information-theoretic irreducibility。

## 2026-11-22 restart accounting
Windows rebooted at 2026-11-20 23:18:55 local; the interrupted processes did not survive. Resumed on 2026-11-22 from saved models and selection steps, with unchanged config and mining pools. A fully frozen dataset now returns before fitting so its original fit timing is preserved. Coveo fit_timing measures the successful resumed invocation only; selection_runtime includes completed saved steps across invocations. Offline time and interrupted unfinished candidate rounds are not silently counted as computation. No test results existed at restart.
