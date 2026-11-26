# Electronics observability and provenance audit

本輪讀取已核對 SHA256 的原始 events.csv，沿用已稽核 local seed42 splits；未宣稱恢復缺失的原 Colab split IDs。先重播原事件 normalization、deduplication、排序與 tie rule，再確認每個歷史 row 的 pre-purchase action sequence、首次購買有無、Act+Rel trace 完全一致。已知數值時間轉為秒；輸入只取可觀察 prefix，再建立 outgoing gap。第20事件的 gap 一律 unknown，不能讀取第21事件時間或隱藏 purchase timestamp。普通零秒 gap 不會當成缺失。

首次檢查在任何模型訓練前發現同一 session key 可對應多位 user，保留失敗紀錄 preparation_attempts.json；所有此類 sessions 一致排除（原 train95、val18、test31），未按模型結果挑選。其餘 session 的 user identity 與歷史 row 一致。重新處理後通過所有 prefix 檢查；準備階段只做數值與資料邊界稽核，沒有 test prediction。

t20 train558/95positive、validation76/8positive、test171/27positive（170users）。保留歷史 session split，因此 train/val 有6位、train/test有12位共同使用者；val/test沒有共同使用者。user-cluster bootstrap 處理 test 內同人 session 相關性，但這不是 unseen-user 泛化實驗。驗證正例只有8個，選模穩定性與推論力度有限，不能因增加第三資料集就聲稱證據普遍增強。

source-edge 使用30分鐘 guard；同一已觀察來源中，未解決 horizon 的右邊界 session 排除（train t20一筆）。source-start 左邊界亦依同一規則處理。短自然終止負例依給定 session IDs 與完整來源推定，這個 closed-session 假設不是部署 censoring 保證。詳見 PROTOCOL.md 及 Electronics/cohort_audit.csv。

舊 master dwell 欄位與由整份 master 求出的 thresholds 均未使用。新 cap與terciles僅由t20training已知gap求出；標準化與raw vocab同樣只fit training。Electronics 11個ActRel propositions給出S272維，T266維；所有缺失標記與額外counts維度見preparation.json。Electronics的p2/p3保留原意：至少2/3個不同產品。不同資料集的抽象語意不能無註解混用。

兩個新候選池都使用同一批2618條training prefix records（t10/20/30僅作mining，不評估額外landmarks），ID/order/labels相同且移除dwell後traces相同。mining設定50epochs/250candidates/seed42，來自Electronics既有budget，learner沿用前輪已驗證的fused-gradient版本。沒有將舊session-purchase候選池冒充future-horizon pool。

獨立特徵驗證：100條實際training prefixes的統計量、Cartesian adjacent pairs、train-only分位數重建與保存特徵一致；改動觀察邊界後的事件/timestamps不影響輸入；人工案例檢查負間隔拒絕、缺失endpoint及terminal mask。neural adapter僅做forward檢查，未增加training fit。所有新test prediction由6LR+9NN的global freeze gate控制。

歷史三資料集dwell來源與舊Electronics洩漏/修補限制見 ../2026-11-23-dwell-time/inventory_zh.md；新資料處理不使舊數字自動變成乾淨或可配對的證據。新研究在已知前兩資料集結果後才新增，保持exploratory標示。
