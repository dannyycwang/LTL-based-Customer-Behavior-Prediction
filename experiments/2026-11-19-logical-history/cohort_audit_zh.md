# Cohort 與標籤稽核

固定 t=10/20/30、m=10、H=10；首次購買事件位置 p 滿足 t<p≤t+10 才是正例，p≤t 排除。保留自然結束且未購買的短負例，不要求存在第 t+10 個事件。所有 A/B/C/D 共用下列 cohorts。

| Dataset | Split | t | Eligible | Positive | Negative | Positive rate | Users | 短負例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Coveo | train | 10 | 56851 | 10055 | 46796 | 17.6866% | 無可靠 ID | 17322 |
| Coveo | train | 20 | 29474 | 6222 | 23252 | 21.1101% | 無可靠 ID | 5771 |
| Coveo | train | 30 | 17481 | 3744 | 13737 | 21.4175% | 無可靠 ID | 2436 |
| Coveo | val | 10 | 12353 | 2255 | 10098 | 18.2547% | 無可靠 ID | 3793 |
| Coveo | val | 20 | 6305 | 1305 | 5000 | 20.6979% | 無可靠 ID | 1211 |
| Coveo | val | 30 | 3789 | 858 | 2931 | 22.6445% | 無可靠 ID | 530 |
| Coveo | test | 10 | 12201 | 2110 | 10091 | 17.2937% | 無可靠 ID | 3763 |
| Coveo | test | 20 | 6328 | 1324 | 5004 | 20.9229% | 無可靠 ID | 1218 |
| Coveo | test | 30 | 3786 | 806 | 2980 | 21.2890% | 無可靠 ID | 565 |
| Cosmetics | train | 10 | 14132 | 4026 | 10106 | 28.4885% | 12049 | 2780 |
| Cosmetics | train | 20 | 7326 | 1985 | 5341 | 27.0953% | 6543 | 968 |
| Cosmetics | train | 30 | 4373 | 1116 | 3257 | 25.5202% | 3991 | 508 |
| Cosmetics | val | 10 | 2027 | 564 | 1463 | 27.8244% | 1743 | 417 |
| Cosmetics | val | 20 | 1046 | 294 | 752 | 28.1071% | 938 | 144 |
| Cosmetics | val | 30 | 608 | 153 | 455 | 25.1645% | 557 | 60 |
| Cosmetics | test | 10 | 4054 | 1193 | 2861 | 29.4277% | 3457 | 768 |
| Cosmetics | test | 20 | 2093 | 580 | 1513 | 27.7114% | 1849 | 309 |
| Cosmetics | test | 30 | 1204 | 286 | 918 | 23.7542% | 1090 | 154 |

Coveo：完整 replay 原 split 的 256,595 sessions；依官方 session timeout 30 分鐘辨識擷取邊界，左端可能缺失的 session 排除，右端未觀察到購買或完整 horizon 者標為截尾。沒有可靠 user ID，bootstrap 以 session 為單位，不能排除同人跨 session 的相關性。為統一檔案 schema，Coveo 個體 CSV 的 `user_id` 欄以 session ID 佔位，不是真實 user ID；彙總 user counts 保留為未知。

事件索引沿用既有 `browsing_train.csv` 事件序列，搜尋不是額外的模型輸入或 horizon 事件。另用同一 ZIP 的完整 `search_train.csv` 補核所有 81,405 個保留 sessions 的活動邊界，以及其中 36,609 個自然短負例的收尾。所有保留 sessions 的最早已知活動距左界至少 65.48 分鐘；短負例的最後已知活動距右界至少 43.90 分鐘，均通過既定 30 分鐘規則。單看 browsing 的最大 session 內間隔為 52.37 分鐘，因此另列較寬邊界的診斷：兩筆 t10 train 短負例距右界約 44／50 分鐘，其中一筆進入 mining；兩者均無搜尋活動，依官方規則已結束。未改 timeout、cohort、pool 或任何模型；沒有跑替代邊界結果。這仍是基於提供者 session 規則及已記錄活動的觀察完整性假設，不是 explicit end-event 證明。詳見 `coveo_edge_margin_audit.json`、`coveo_search_closure_audit.json`。

Cosmetics：完整 replay 原 December split 的 130,117 sessions。原始 session ID 不能可靠套用 30 分鐘 timeout；另核對同一官方 release 的其他四個月份，1,066 個 December session keys 出現在其他月份，排除其 prefixes。保留 session 依原 ID 與相鄰月份沒有延續紀錄推定自然結束；此假設不等於有 explicit session-end event，仍可能有記錄缺漏。沒有用暫議的月末 24 小時 guard。額外月份只作邊界核對，沒有新增訓練／測試 cohort。

沿用原 split，驗證 session 不跨 split；Cosmetics 再驗證 user 不跨 split。不同 t 的 prefixes 繼承 session split。資料曾經過歷史 cohort 抽樣，本輪 AP/Brier 不能外推為自然流量下的 calibration。

Cosmetics 沿用既有負價記錄清理。另對原始 December 全檔稽核：全檔有 18 筆負價事件；本輪 eligible sessions 聯集中有 10 個 sessions 含 11 筆負價 purchase 記錄。逐 session 比對原始與清理後序列，確認所有首次購買前的事件序列與首次購買事件位置均相同，因此本輪 cohort、prefix 特徵與 future-first-purchase 標籤均不受該清理影響。稽核保存於 `price_cleaning_audit.json`，最終驗證會檢查此項結果。

每欄排除理由與原始 hash/replay 細節保存於各 dataset 的 cohort_audit.csv、raw_audit.json、preparation.json 與 boundary_followup.json（若適用）。

特徵稽核：未來 suffix mutation 不改任何輸入、t10 B/C 全部輸入一致、900 個實際 prefixes 的摘要獨立重建相同；mining IDs/labels 僅取 training split。公式 satisfaction 的獨立 finite-trace checker 與正式 t10 identity 另於軟體／最終驗證檢查。
