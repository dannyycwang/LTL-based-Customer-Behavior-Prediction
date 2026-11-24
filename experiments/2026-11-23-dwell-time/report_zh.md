# 固定 dwell 補充實驗結果

## 完成後核對：改稿應採用的結論

一般時間特徵的收益有資料集差異。線性 M2−M0 的 AP 在 Coveo 為 +0.01658（95% 區間 +0.00604 至 +0.02678），支持此固定設定下的改善；Cosmetics 為 +0.00505（−0.00983 至 +0.02073），尚無明確證據。兩者的 MCC 與 Brier 差值區間均跨零；Cosmetics 的 Brier 點估計反而增加 0.00313，因此不能宣稱所有指標一致改善。

在已有 R+S+T 後，兩資料集的線性 M4−M3、M4−M2、M4−M5 AP 區間全部跨零；神經三 seed 平均的 M4−M3 與 M4−M2 AP 區間也全部跨零。本輪沒有證實 dwell vocabulary 的 learned LTLf 額外 AP 增益，也沒有證實 learned LTLf 優於 DECLARE。這不是等價或無效果的證明。

神經結果需保留次要指標與 seed 差異。Cosmetics 的 M4−M3 平均 Brier 差為 −0.01345（−0.01563 至 −0.01112），但三個 seeds 分別為 −0.03901、−0.00723、+0.00590，並非逐 seed 一致改善。其 AP 差值分別為 +0.01171、+0.01147、−0.02154，平均 +0.00055、sample SD 0.01913。Coveo 的對應 AP 差值為 −0.00340、−0.00063、+0.00060，平均 −0.00114、sample SD 0.00205。所有六個單 seed 主要 AP 差值區間也都跨零。固定三個模型的 cluster bootstrap 不包含 optimization-seed 變異，不能將較窄的平均區間當成跨訓練隨機性的穩健證據。

Cosmetics 神經 M4−M2 的 MCC 改善 +0.05156（+0.02670 至 +0.07667），Brier 改善 −0.01036（−0.01266 至 −0.00821）；M3−M2 的 MCC 也改善 +0.04241（+0.02232 至 +0.06515），但 Brier 變差 +0.00308（+0.00144 至 +0.00463）。這些探索性次要結果應完整保留，不能替代事前指定的 primary AP 結論，亦不構成架構普遍優越的證據。

M4 的 10 個公式中，兩資料集各有 4 個涉及已知間隔 buckets；Cosmetics 另有 2 個僅涉及未知標記 du 的時間 vocabulary。所有公式都有 temporal-operator 語法。20 個 M4 公式皆未與本次枚舉的單一 dwell-count 門檻規則在 training 上完全相同，但 Coveo 一個含 bucket 公式的最佳一致率達 99.9016%。這個有限診斷既不證明非冗餘，也不證明公式提供無法由強摘要表達的資訊；高一致率也可能受 support 不平衡影響。

選擇成本差異很大：Coveo M3/M4/M5 分別約 3.53/4.35/19.91 小時，Cosmetics 約 0.22/0.87/7.91 小時；這是當次候選重擬合的 wall time，含同機資源競爭。相同十特徵上限並未使搜尋成本相同。不得把後續快速 GPU 訓練解釋成整套方法端到端加速。

改稿宜保留「可檢查的邏輯表示」與精確的條件式結果；將「dwell/LTLf 普遍提升預測」縮小為本輪未獲 primary AP 支持的探索性比較。保留 Cosmetics 的次要指標改善及其 seed 限制；刪除時間間隔代表注意力、單一公式因果作用、跨任務直接比較與整體訓練加速等未被此實驗支持的說法。


本輪固定矩陣已完成：兩資料集六個線性條件，加上 M2/M3/M4 各三個神經 optimization seeds，共18個神經 fits。舊 test 曾參與研究開發，所有結論均為 exploratory。時間變數是已完成 event gap，不是真實注意力時間。

## 直接回答研究問題

### Coveo

- 一般時間特徵（M2-M0）：+0.01658 [+0.00604, +0.02678]；固定模型的 AP 差值區間完全在零以上，但仍是探索性證據。
- 含 dwell vocabulary 的 learned LTLf 相對 Act+Rel LTLf（M4-M3）：+0.00093 [-0.00203, +0.00375]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 已提供普通時間特徵後的 LTLf 增益（M4-M2）：+0.00221 [-0.00032, +0.00457]；AP 差值區間跨零，未證實額外增益；不代表等價。
- learned LTLf 相對 DECLARE（M4-M5）：-0.00005 [-0.00646, +0.00576]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 神經模型主要對比（三 seed 平均 M4−M3）：-0.00114 [-0.00454, +0.00252]；AP 差值區間跨零，未證實額外增益；不代表等價。各 seed 與 mean/sample SD 見附表，三 seed 不當成三倍樣本。

線性與神經主要對比的點估計方向不同，不支持跨模型一致的增益主張。

- 神經三 seed 平均 M4-M2：-0.00195 [-0.00800, +0.00415]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 神經三 seed 平均 M3-M2：-0.00081 [-0.00722, +0.00565]；AP 差值區間跨零，未證實額外增益；不代表等價。

主要對比 M4−M3 的次要指標如下（MCC 越高越好，Brier 越低越好；不依方向挑選報告）：

- 線性：MCC -0.00645 [-0.01319, +0.00037]；Brier +0.00007 [-0.00032, +0.00043]。
- 神經三 seed 平均：MCC -0.00117 [-0.00603, +0.00384]；Brier -0.00012 [-0.00063, +0.00040]。
M4 選中 10 個公式，其中 4 個含時間 vocabulary，4 個含真正已知間隔 buckets，0 個含未知 marker du，10 個含 temporal operator 語法。du 在本資料也是末端標記，不能將只用 du 的公式當成 duration 效應。語法不等於不可替代的時序資訊；training count-rule 冗餘和 support 見 selected_formulas.csv。
### Cosmetics

- 一般時間特徵（M2-M0）：+0.00505 [-0.00983, +0.02073]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 含 dwell vocabulary 的 learned LTLf 相對 Act+Rel LTLf（M4-M3）：+0.00108 [-0.00347, +0.00500]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 已提供普通時間特徵後的 LTLf 增益（M4-M2）：+0.00125 [-0.00332, +0.00522]；AP 差值區間跨零，未證實額外增益；不代表等價。
- learned LTLf 相對 DECLARE（M4-M5）：+0.00105 [-0.00597, +0.00787]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 神經模型主要對比（三 seed 平均 M4−M3）：+0.00055 [-0.00867, +0.00948]；AP 差值區間跨零，未證實額外增益；不代表等價。各 seed 與 mean/sample SD 見附表，三 seed 不當成三倍樣本。

至少一種模型的主要對比未證實正向增益，不能宣稱神經結果確認了線性模型的收益。

- 神經三 seed 平均 M4-M2：+0.00631 [-0.00271, +0.01565]；AP 差值區間跨零，未證實額外增益；不代表等價。
- 神經三 seed 平均 M3-M2：+0.00576 [-0.00254, +0.01442]；AP 差值區間跨零，未證實額外增益；不代表等價。

主要對比 M4−M3 的次要指標如下（MCC 越高越好，Brier 越低越好；不依方向挑選報告）：

- 線性：MCC +0.01717 [-0.00305, +0.03524]；Brier -0.00046 [-0.00168, +0.00087]。
- 神經三 seed 平均：MCC +0.00915 [-0.00929, +0.02705]；Brier -0.01345 [-0.01563, -0.01112]。
M4 選中 10 個公式，其中 6 個含時間 vocabulary，4 個含真正已知間隔 buckets，2 個含未知 marker du，10 個含 temporal operator 語法。du 在本資料也是末端標記，不能將只用 du 的公式當成 duration 效應。語法不等於不可替代的時序資訊；training count-rule 冗餘和 support 見 selected_formulas.csv。

## 主張與引用範圍

保留：在此固定 cohort、觀察邊界和模型下觀察到的 AP/MCC/Brier 數值與配對差值；邏輯特徵可被明確檢查。縮小：任何收益只能歸於整套 vocabulary／學習／選擇流程，不能歸因單一 dwell operator。神經結論限既有 encoder/fusion 與 LR 移用公式；不同模型結果不得互相替代。刪除或避免：真實注意力、因果機制、普遍 superiority、未顯著即等價、整體訓練加速，以及以舊 session-purchase 成績支持新固定 landmark 任務的主張。

舊三資料集 dwell 結果的可引用性見 inventory_zh.md。Electronics 原版確定洩漏；修補版仍有 master thresholds／未知零值混同限制。Coveo 舊 short-window 邊界與 median 填補不同；Cosmetics 舊 parser／purchase-relative task 不同。歷史彙總可註明限制後引用為背景，不能與本輪兩資料集新結果配對。

## 完整結果與限制

| dataset   | condition   |       AP |    Brier |      MCC |   selected_logic_features |
|:----------|:------------|---------:|---------:|---------:|--------------------------:|
| Cosmetics | M0          | 0.404481 | 0.207781 | 0.182219 |                         0 |
| Cosmetics | M1          | 0.404974 | 0.207691 | 0.172627 |                         5 |
| Cosmetics | M2          | 0.409534 | 0.210913 | 0.213567 |                         0 |
| Cosmetics | M3          | 0.409709 | 0.210732 | 0.202495 |                         8 |
| Cosmetics | M4          | 0.410786 | 0.21027  | 0.219662 |                        10 |
| Cosmetics | M5          | 0.40974  | 0.211012 | 0.204988 |                        10 |
| Coveo     | M0          | 0.505594 | 0.127658 | 0.468111 |                         0 |
| Coveo     | M1          | 0.509207 | 0.127164 | 0.467374 |                        10 |
| Coveo     | M2          | 0.522178 | 0.126446 | 0.464535 |                         0 |
| Coveo     | M3          | 0.523457 | 0.126201 | 0.46658  |                        10 |
| Coveo     | M4          | 0.524391 | 0.126271 | 0.460127 |                        10 |
| Coveo     | M5          | 0.524442 | 0.125902 | 0.466359 |                        10 |

AP 為 primary；不得忽略 metrics.csv 中的 MCC 與 Brier，完整 paired intervals 在 paired_differences.csv。所有 interval 固定模型、公式及 thresholds，不涵蓋 mining uncertainty；沒有多重比較校正。Cosmetics 按 user，Coveo 按 session，後者不能排除同人跨 session 相關性。樣本為歷史 cohort，非自然部署 prevalence。

時間成本分項見 runtime.csv／selection_cost.csv；selection 含候選分類器重擬合，不重複相加；cached-input neural training 不是端到端速度。所有個體資料、IDs、預測與模型在 private。原論文與既有實驗未覆寫。本輪矩陣完成後停止，未自動延伸研究。

針對既有恢復版原稿的逐項改稿位置、時間定義修正與英文 Methods/limitations 替換文字，見 [claims_revision_map_zh.md](claims_revision_map_zh.md)。該來源不是已核對的最新投稿原稿，未直接覆寫論文。

神經配對差值的跨 seed mean/sample SD 另見 neural_paired_mean_sd.csv。這是三個 optimization seeds 的變異描述，與共用 user/session resamples 的固定模型 bootstrap 區間不同；後者不估計所有可能 optimization seeds 的不確定性。