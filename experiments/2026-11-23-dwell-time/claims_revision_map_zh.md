# 改稿主張對照與時間定義修正

對照來源為 `../2026-11-17/paper_recovered_revision/main.tex`，是既有「恢復的較早 LaTeX 版本」，不是已確認的最新投稿原稿。本檔提供可移用的修改方向，不覆寫原稿。新實驗尚未完成時，不填寫或猜測其數值結論；數值請以完成且驗證通過的 report_zh.md 與 manuscript_results_en.md 為準。

| 對照位置 | 處理 | 可移用的修正方向 |
|---|---|---|
| Abstract，line134；Conclusion，line527 | 保留架構描述，縮小效益主張 | 保留「有限軌跡的明確 Boolean 公式特徵＋短 raw 序列」。舊 event10／Counts 對照和新 event20／強摘要＋時間對照分開報告；不能將舊 count-augmented LR 收益泛稱為超越所有普通摘要，也不能由線性結果推論神經融合收益。 |
| Introduction，line161；representation，lines182、269 | 必須修正時間可觀察性 | event gap 不是真實注意力或獨立記錄的停留時間。outgoing gap 不會在該事件剛發生時完整可觀察；在 prediction origin t，只能用 j<t 的完成間隔，第t格未知。不能把 action、product/category 和完整 outgoing gap 無條件視為同時取得的 event attributes。 |
| Abstraction example，line307 | 保留例子但加上時間索引條件 | 已完成的較早 detail event 可以具有 long-gap proposition；當前末端 detail event 只能是 unknown-gap。關係命題可依 prefix 因果建立，但這不自動保證 timing proposition 也符合可觀察性。 |
| Fixed K representation，line187 | 保留數學性質，限制解讀 | K維輸出不隨序列長度變動是表示性質；不代表保留所有相關資訊、常數時間計算、低總訓練成本或因果可解釋性。語意明確與預測重要性是不同主張。 |
| Vocabulary extension，line305 | 保留機制，補上比較限制 | 擴 vocabulary 可沿用相同 LTLf satisfaction checker；M4−M3 比較的是兩套 vocabulary／mining／selection 流程，不能稱作單一 dwell operator 的因果效果。du 亦可能充當末端標記，須與已知間隔 buckets 分開診斷。 |
| Neural/linear interpretation，line450 | 縮小至實際比較 | 舊神經對照與舊 LR 對照不是同一條件差。新18個 fits 使用同一 S+T 與 raw timing，才回答這一固定設定下 M3/M4 的增益；LR選出的公式移用神經模型仍是限制。未顯著不能稱等價。 |
| Horizon analysis，line464 | 保留舊結果標籤，不拼接 | 舊 H5/10/20 結果保留在舊研究；新矩陣只有 t20、m10、H10，不能混用不同 cohort 的 AP 或把它們當 paired comparisons。新研究只有 Coveo/Cosmetics，Electronics 只做歷史稽核。 |
| Cost section，lines514–518 | 保留原計時邊界，刪除端到端外推 | 舊 cached-input update/GPU allocation 數值只能在原 benchmark 條件引用。新實驗另列 preprocessing、mining、selection、classifier training、inference，selection含候選重擬合。不得把 cached-input training 或有限K當成整體加速；本輪未新增 Full raw benchmark。 |
| Discussion/Conclusion，lines524–527 | 依完成後結果重寫 | 對 AP、MCC、Brier 的結果完整陳述。只有實際支撐的 dataset/model/contrast 可保留狹義效益敘述；一般 superiority、自動公式學習必然勝過 DECLARE、注意力／購買機制的因果推論均不成立。先前提出的後續研究不在本輪自動執行。 |

## 可直接移用的英文 Methods／limitations 文字

At prediction event \(t=20\), we first truncate each session to its observable prefix. We define an outgoing event gap for event \(j<t\) as \(g_j=(\tau_{j+1}-\tau_j)/1000\) seconds, where \(\tau\) is the timestamp in milliseconds. The complete outgoing gap of event \(t\) is not yet observable and is explicitly masked. A genuine zero-second gap remains distinct from an unknown gap. These timestamp differences are event-gap proxies, not measurements of attention. Bucket thresholds and feature transformations are fitted on training prefixes only. The same timing inputs are supplied to all timing-augmented conditions.

The primary contrast compares Act+Rel+Dwell and Act+Rel formula-learning pipelines conditional on the same raw, strong-summary, and nonlogical timing features. It does not identify the causal effect of an individual dwell operator. The unknown-gap proposition may also mark the observable trace endpoint; formulas using it alone are distinguished from those involving known-duration buckets. Formulas selected using logistic regression are transferred unchanged to the neural comparison. All results are exploratory because the test set has participated in research development, and the paired intervals condition on fixed models and formulas rather than accounting for mining uncertainty.
