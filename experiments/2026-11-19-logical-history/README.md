# 邏輯歷史記憶：固定 landmark 探索性 pilot（2026-11-19）

本輪只比較 logistic regression 的 A/B/C/D，primary 為 t=20 的 AP(C)−AP(A)。既有 test 已參與開發；所有結果都是探索性。原論文、既有實驗、GitHub remote 均不修改。

已於 2026-11-23 完成全部六個 cohorts 的選擇、凍結、test 評估與 1,500 次 paired bootstrap，最終一致性驗證通過。研究判斷：**證據不足，只值得補一項最關鍵的檢查。** 見 [繁體中文報告](report_zh.md)、[主要結果表](main_results.csv) 與 [paired AP 圖](paired_ap_differences.png)。本輪不再自動延伸實驗。

## 執行與來源

從 repository 的 `experiments/2026-11-19-logical-history` 執行：

```powershell
.\run.ps1
```

入口會在隔離目錄安裝固定的 `liblinear-official==2.50.0`（若尚未安裝），依序準備兩個資料集、稽核、建立／重用唯一 pools、選擇與凍結全部模型，最後評估、繪圖與執行最終一致性驗證。個別階段亦可用 `python -u pilot.py prepare <dataset>`、`python -u mine.py <dataset>`、`python -u fit.py fit <dataset>`、`python -u fit.py evaluate <dataset>` 呼叫；evaluation 會拒絕任何尚未全部凍結的狀態。

`config.json` 固定所有研究參數與原 private project 位置。沿用原環境 Python 3.10、`work/python_packages` 中的 sklearn 1.7.2，以及原 audit modules；原始資料不隨 GitHub 發布。`pilot.py` 會驗證原 split / dataset / basis hashes。`vendor/` 保存實際使用的 TLTLf learner，原檔不變；移除無用 TensorBoard logging，其餘使用先前已驗證的 CPU fused-gradient 實作。新機器需重建 `config.json` 的 private source layout；這是與既有研究環境相容的重跑程式，不宣稱是免資料依賴套件。`mine.py` 對已完成的單一 pool 直接重用；`fit.py` 對已凍結模型重用。完整重跑應將本輪程式與 config 複製到新的輸出目錄，不覆寫既有結果。

檢查時 repository HEAD：`00f7c186c17748e842eb6fdb6b95d00a9e1c802c`。最新 README 指向 2026-11-17 兩輪實驗，沒有本輪三個 landmarks / 強摘要 / recent-window 對照。舊 pools 使用 eventual-purchase 或 purchase-relative boundaries，不符合本輪 future labels，故不沿用。

## 任務與資料邊界

- t=10/20/30，m=10，H=10；先觀察第 t 個事件，再預測 t+1..t+10 是否首次購買；購買位置 ≤t 排除。原始事件排序沿用 timestamp、原檔 row order 的 stable tie-breaking。
- Coveo 的事件時鐘沿用 browsing log 的 pageview / detail / add / remove / purchase；不混入獨立 search log。Cosmetics 沿用原清理後 event log（缺失時間/key、負價格處理與舊程式一致）。這些是既有研究的 event universe，並非跨所有網站事件的時鐘。
- Coveo 全部 256,595 個 split sessions 的 raw label / first-purchase 前 action 序列 replay 一致。原始擷取左端 30 分鐘內開始的 session 排除；右端 30 分鐘內尚未完成 horizon 且無購買的 session 視為 censored。其餘自然結束負例不要求有 t+10 個事件。30 分鐘規則來自 [Coveo 官方資料定義](https://github.com/coveooss/SIGIR-ecom-data-challenge)。實際記錄少數 gap 超過 30 分鐘，可能有 tracking inconsistency，原始核對檔揭露最大值；不能宣稱日志沒有漏失。
- Cosmetics 先 replay 全部 130,117 個原 pre-purchase split sessions。另從 [原 Kaggle 來源](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop/data) 下載其他四個月份，僅核對 `(user_id,user_session)`；任何出現在別月的原 session 一律排除，不改 split、不重建較長的新 session、不加入新月份 cohort。保留的 session 以原 ID 與完整相鄰月份沒有延續紀錄推定結束；沒有 explicit end event，仍有未記錄活動的限制。最初提及的 24 小時 guard 未獲採用，也未用於任何訓練或評估。
- 同一 session 的所有 prefixes 繼承原 split；Cosmetics 額外驗證 user-disjoint。Coveo 沒有 user ID，不能驗證 user-disjoint 或進行 user bootstrap。

## 四條件與公式選擇

Proposition 定義、缺失 ID 處理與有限軌跡端點語義見 `propositions.md`。完整 cohort 表及負價記錄不影響本輪 prefixes／首次購買位置的驗證見 `cohort_audit_zh.md`、`price_cleaning_audit.json`。

A = 最近十個 action/product/category positional one-hot（train-only vocabulary）＋前 t 個事件摘要：每個 proposition count/presence、全部有向相鄰 proposition pairs 的 occurrence 與 frequency（次數/(t−1)）、四個 actions 的 recency 與 never bit。proposition pair 是相鄰兩個 proposition sets 的 Cartesian product，可同時含多個 pair。不存在 Top-10 transition 篩選。未出現 action 的 recency 設為 t，另有明確 never bit。

B = A＋在最近十個事件**重新初始化** product/category 記憶後的 learned LTLf；C = A＋前 t 個事件的 learned LTLf；D = A＋前 t 個事件上的既有 DECLARE existence/response/precedence/chain-response 模板。B 不繼承較早的 repeat-product 等狀態。A/C/D 使用完整已觀察 prefix 的抽象；因此 C−B 同時反映觀察範圍、抽象與特徵選擇差異，不是純公式長距離能力。

每個資料集使用單一 seed=42 pool。從三個 train landmarks 等額、按 future label 分層、無放回抽樣；每 class 總預算沿用 Coveo 10,000、Cosmetics 5,000，除以三向下取整，某 stratum 不足則全取，不從別的時點補量。完整 1..t prefixes 進 learner，同一 session 可貢獻不同 t 的樣本，但不跨 split。實際數量在 `mining/sampling_counts.csv`。Coveo 100 epochs、200 candidates；Cosmetics 60 epochs、300 candidates；兩者 netlen=10、AdamW lr=.01、batch=100、既有 .1 regularization、train-loss checkpoint。只保留這一個 seed，沒有額外候選補入或 test ranking。

Selector 改自 `../2026-11-17-residual-selection/run_residual.py::select`：保留 greedy validation log loss、重新擬合、候選次序 tie-breaking。修改為以完整 A 為 baseline、sparse raw encoding、最多十個、gain≤1e−10 時停止。移除 training-constant 與 exact training-duplicate candidate columns，也排除等於 A 既有 binary 欄的 candidates。每一步保存全部候選 loss/gain、候選數、公式與 wall time。原模型使用的 A 摘要欄固定保留。

數值 scaling 只 fit train；raw one-hot 不 standardize，與既有 logistic pipeline 相同。固定 C=1、L2-regularized primal logistic regression（intercept 同樣懲罰）。執行版本使用官方 `liblinear-official==2.50.0`，candidate 以上一步模型初始化，eps=1e−6；每次以解析 gradient 檢查收斂，失敗即停止。此為求解器實作差異：繼承 sklearn 版本使用冷啟動、eps=1e−4、max_iter=2000；原生版本內部上限為 1000，沒有暴露 iteration count。四條件統一使用新版本；未完成的舊選擇封存在 `execution_archive`，沒有混入本輪結果。固定的數值比較門檻與結果見 `native_check.json`，不代表數值上完全相同，也不以 validation AP 或 test 決定求解器。MCC threshold 僅由 validation 固定 grid .001, .01.. .99, .999 選擇。所有 requested datasets/landmarks 的模型與公式凍結後才允許 test evaluation。t=10 在 train/validation/test 逐候選驗證 B=C，並重用相同 selection，避免重跑等價模型。

## 不確定性與輸出

1,500 次 paired cluster bootstrap，共用每次的抽樣權重；Cosmetics 按 user，Coveo 按 session。AP/MCC/Brier 與 C−A/C−B/C−D 均輸出 point estimates 和 marginal percentile 95% intervals。不計 p-values；intervals 固定模型、公式、threshold，不含 mining/selection uncertainty，也沒有 multiple-comparison correction。不同 t 是不同 eligible cohorts，跨 t 差異不能當成增加歷史長度的因果效果。

每個 dataset 保存 cohort CSV、來源與 split hashes、原事件 replay、training sample IDs、候選 pool、選定公式及逐步 traces、凍結模型、共同 test predictions、metrics / paired differences CSV。最終彙整為 `main_results.csv`、`metrics.csv`、`paired_differences.csv`、`paired_ap_differences.png` / `.svg`、繁體中文 `report_zh.md`。個體資料、下載的原檔與模型列在 `.gitignore`，不推送 GitHub。

實際 wall time 包含同機並行工作的影響，不是硬體 benchmark；沒有串流監測器、memory/speed superiority 或 neural generalization 主張。
