# 三資料集 dwell 補充比較

## 結果綜合判讀

新增 Electronics 後，三資料集的 LR 與三 seed 平均神經模型共六項主要 M4−M3 AP 比較，其 95% 配對區間皆跨零。這些實驗未提供 dwell vocabulary 穩定增加 AP 的證據，也不構成等價證明。Electronics 的 LR AP 差為 −0.03988 [−0.12317, +0.02018]；神經平均為 −0.00125 [−0.00585, +0.00359]。
Electronics 的次要結果也不支持普遍提升：LR M2−M0 的 MCC 下降 −0.16438 [−0.30918, −0.00570]；LR M4−M2 的 Brier 上升 +0.01445 [+0.00231, +0.02828]。神經 M4−M3 的平均 MCC 下降 −0.11741 [−0.17797, −0.04581]，其中 seed42 的差異為 −0.32967、seed43 為 −0.02255、seed44 為 0，效果集中於一個 seed。
神經 M4−M2 的平均 Brier 改善 −0.01602 [−0.02227, −0.01030]，但三 seed 差異為 +0.01486、−0.01806、−0.04487，方向不一致；M3−M2 的平均改善甚至略大（−0.01660）。故這項改善不能專屬歸因於 dwell vocabulary。MCC 使用各模型在僅 8 個正例的 validation set 選出的門檻，應連同 AP、Brier 與 seed 表解讀。
M4 的 10 個公式中 7 個含已知間隔 bucket、0 個含 du、9 個含 temporal operator；單獨原子 dl 也是選中公式，因此不應聲稱全部都是時間運算子規則。沒有與本輪枚舉的單一 dwell-count 門檻完全相同者，但此診斷不排除其他冗餘。
兩次候選池挖掘各約 436 與 513 秒；四組選擇合計約 305 秒，含 6,260 次候選 refits。成本是本機測量，不能由 cached-input 神經訓練時間推論端到端加速。

Coveo、Cosmetics 沿用 2026-11-23 完成並凍結的結果；Electronics 是看過前兩組結果後，依使用者於 2026-11-26 要求新增的探索性擴充，不是原先事前固定範圍。沒有重跑或改選前兩資料集。新實驗仍只評估 t=20、m=10、H=10。

## 主要 AP 比較

| dataset     | model   |        delta |    lower_95 |   upper_95 |
|:------------|:--------|-------------:|------------:|-----------:|
| Coveo       | LR      |  0.000933434 | -0.00203081 | 0.00375489 |
| Coveo       | NN_mean | -0.00114463  | -0.0045378  | 0.00251506 |
| Cosmetics   | LR      |  0.00107698  | -0.00346589 | 0.00499986 |
| Cosmetics   | NN_mean |  0.000546562 | -0.00867496 | 0.00947861 |
| Electronics | LR      | -0.039879    | -0.123173   | 0.0201829  |
| Electronics | NN_mean | -0.00125053  | -0.0058544  | 0.00358608 |

Electronics 的 t20 cohort 為 train558（95正例）、validation76（8正例）、test171（27正例，170 users）。樣本與驗證正例數很少，選模及效果估計可能不穩定；第三資料集不能自動當作更強或普遍的證據。

## Electronics 直接回答

- LR M2-M0 AP：-0.02054 [-0.15995, +0.11499]；區間跨零 / interval includes zero。
- LR M4-M3 AP：-0.03988 [-0.12317, +0.02018]；區間跨零 / interval includes zero。
- LR M4-M2 AP：-0.03308 [-0.09547, +0.00809]；區間跨零 / interval includes zero。
- LR M4-M5 AP：+0.02098 [-0.03351, +0.08139]；區間跨零 / interval includes zero。
- NN_mean M4-M3 AP：-0.00125 [-0.00585, +0.00359]；區間跨零 / interval includes zero。
- NN_mean M4-M2 AP：+0.01084 [-0.05288, +0.07017]；區間跨零 / interval includes zero。
- NN_mean M3-M2 AP：+0.01209 [-0.04998, +0.07043]；區間跨零 / interval includes zero。

M2−M0 回答普通時間特徵；M4−M3 回答擴充 learned vocabulary 流程；M4−M2 回答已有 T 後的額外邏輯收益；M4−M5 比較 learned LTLf 與 DECLARE。區間跨零不代表等價；不可將單一 operator 或事件間隔視為因果、注意力或普遍架構效益。

## 完整次要指標與 seed 變異

| dataset     | model   | contrast   | seed   | metric   |        delta |     lower_95 |    upper_95 | unit   |   units |   replicates |   valid_replicates |
|:------------|:--------|:-----------|:-------|:---------|-------------:|-------------:|------------:|:-------|--------:|-------------:|-------------------:|
| Electronics | LR      | M4-M3      | 42     | MCC      | -0.181729    | -0.349885    |  0.0101719  | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M3      | 42     | AP       | -0.039879    | -0.123173    |  0.0201829  | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M3      | 42     | Brier    |  0.0108211   | -0.00527371  |  0.0274915  | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M2-M0      | 42     | MCC      | -0.164379    | -0.309177    | -0.00569851 | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M2-M0      | 42     | AP       | -0.0205425   | -0.159953    |  0.114994   | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M2-M0      | 42     | Brier    |  0.0285363   | -0.000123082 |  0.0558108  | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M2      | 42     | MCC      |  0.051031    | -0.137635    |  0.249358   | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M2      | 42     | AP       | -0.0330796   | -0.0954731   |  0.00808508 | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M2      | 42     | Brier    |  0.0144534   |  0.00231358  |  0.028283   | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M5      | 42     | MCC      | -0.0779585   | -0.184292    |  0          | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M5      | 42     | AP       |  0.0209848   | -0.033511    |  0.081387   | user   |     170 |         1500 |               1500 |
| Electronics | LR      | M4-M5      | 42     | Brier    | -0.00369298  | -0.0244836   |  0.0152551  | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M3      | mean   | MCC      | -0.117409    | -0.177974    | -0.0458106  | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M3      | mean   | AP       | -0.00125053  | -0.0058544   |  0.00358608 | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M3      | mean   | Brier    |  0.000581432 | -4.63221e-05 |  0.00121263 | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M2      | mean   | MCC      | -0.0966969   | -0.232919    |  0.0485     | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M2      | mean   | AP       |  0.0108436   | -0.0528776   |  0.0701716  | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M4-M2      | mean   | Brier    | -0.0160229   | -0.0222733   | -0.0103035  | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M3-M2      | mean   | MCC      |  0.020712    | -0.0841108   |  0.120514   | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M3-M2      | mean   | AP       |  0.0120941   | -0.0499816   |  0.0704278  | user   |     170 |         1500 |               1500 |
| Electronics | NN_mean | M3-M2      | mean   | Brier    | -0.0166043   | -0.0230252   | -0.0107755  | user   |     170 |         1500 |               1500 |

| dataset     | model   | contrast   |   seed | metric   |        delta |     lower_95 |     upper_95 | unit   |   units |   replicates |   valid_replicates |
|:------------|:--------|:-----------|-------:|:---------|-------------:|-------------:|-------------:|:-------|--------:|-------------:|-------------------:|
| Electronics | NN      | M4-M3      |     42 | MCC      | -0.329674    | -0.495044    | -0.124523    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     42 | AP       | -0.00170274  | -0.00622748  |  0.00343923  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     42 | Brier    |  0.00050608  |  6.35475e-05 |  0.000926167 | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     43 | MCC      | -0.022553    | -0.085114    |  0.0126142   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     43 | AP       |  0.000367192 | -0.00232626  |  0.00389989  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     43 | Brier    |  0.000631324 |  0.000258864 |  0.00099331  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     44 | MCC      |  0           |  0           |  0           | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     44 | AP       | -0.00241605  | -0.0134463   |  0.00939076  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M3      |     44 | Brier    |  0.000606892 | -0.000840862 |  0.00213446  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     42 | MCC      | -0.314742    | -0.47316     | -0.115615    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     42 | AP       | -0.00595445  | -0.0825151   |  0.0686195   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     42 | Brier    |  0.014858    |  0.00344335  |  0.0262272   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     43 | MCC      |  0.0272648   | -0.0293762   |  0.10258     | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     43 | AP       |  0.00386744  | -0.087091    |  0.0865305   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     43 | Brier    | -0.018056    | -0.0273733   | -0.0088643   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     44 | MCC      | -0.00261329  | -0.28976     |  0.272359    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     44 | AP       |  0.0346178   | -0.120063    |  0.153696    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M4-M2      |     44 | Brier    | -0.0448707   | -0.0684984   | -0.0212592   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     42 | MCC      |  0.0149315   | -0.0706834   |  0.0998758   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     42 | AP       | -0.00425172  | -0.0803102   |  0.0689487   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     42 | Brier    |  0.0143519   |  0.00309707  |  0.0256199   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     43 | MCC      |  0.0498179   | -0.0265834   |  0.148541    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     43 | AP       |  0.00350024  | -0.0872378   |  0.0835239   | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     43 | Brier    | -0.0186873   | -0.0280527   | -0.00951557  | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     44 | MCC      | -0.00261329  | -0.28976     |  0.272359    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     44 | AP       |  0.0370338   | -0.115095    |  0.155614    | user   |     170 |         1500 |               1500 |
| Electronics | NN      | M3-M2      |     44 | Brier    | -0.0454776   | -0.0686954   | -0.0223715   | user   |     170 |         1500 |               1500 |

| dataset     | contrast   | metric   |         mean |         std |
|:------------|:-----------|:---------|-------------:|------------:|
| Electronics | M3-M2      | AP       |  0.0120941   | 0.0219434   |
| Electronics | M3-M2      | Brier    | -0.0166043   | 0.0299691   |
| Electronics | M3-M2      | MCC      |  0.020712    | 0.0266893   |
| Electronics | M4-M2      | AP       |  0.0108436   | 0.0211666   |
| Electronics | M4-M2      | Brier    | -0.0160229   | 0.0299162   |
| Electronics | M4-M2      | MCC      | -0.0966969   | 0.189423    |
| Electronics | M4-M3      | AP       | -0.00125053  | 0.00144568  |
| Electronics | M4-M3      | Brier    |  0.000581432 | 6.63905e-05 |
| Electronics | M4-M3      | MCC      | -0.117409    | 0.184172    |

## 公式診斷與資料限制

M4 選中 10 個公式；7 個含已知間隔 bucket，0 個含 du，9 個含 temporal 語法；0 個與枚舉的單一 dwell-count 門檻規則在 training 完全相同。有限診斷不證明非冗餘，詳見 selected_formulas.csv。Electronics p2/p3 表示不同產品數，不是目前產品重複次數。
保留原 session split；t20 使用者跨 split 重疊數：{'train_val': 6, 'train_test': 12, 'val_test': 0}。按 user bootstrap 處理 test 內相關性，不能修補 train/test 使用者重疊。本輪不是 unseen-user 泛化測試。已在訓練前排除同一 session key 對應多個 user 的歸屬不明 sessions；排除數列於 cohort_audit.csv。原始 source-edge 使用30分鐘 guard，短自然終止負例依給定 session ID 與已觀察歷史判定；不聲稱自然部署或所有 censoring 條件。
間隔先 slice 再計算，末端 unknown；零秒另行保留，buckets/scalers 僅 training 擬合。舊 Electronics master dwell 與 buckets 未使用。匹配 mining 使用 train t10/20/30 記錄但不新增這些 evaluation landmarks。舊 session-purchase 成績與本輪 future-horizon 結果不可直接配對。

## 改稿與時間成本

保留三資料集各自的 AP/MCC/Brier 與配對區間；將神經結論限於固定 encoder、融合方式和 LR 移用公式。縮小或刪除普遍提升、單一 dwell 因果作用、真實注意力、未顯著即等價與端到端加速等主張。歷史三資料集稽核見前輪 inventory；本輪三資料集新固定-landmark 結果另列，不能混為舊任務重現。
成本見 runtime.csv 與 selection_cost.csv：preprocessing、兩次匹配 vocabulary mining、selection、baseline classifier、neural training、inference 分列；selection 已含候選分類器重擬合不可重複加總。cached-input neural training 不等於整體速度。

## 三資料集表

| dataset     | condition   |       AP |    Brier |      MCC |
|:------------|:------------|---------:|---------:|---------:|
| Cosmetics   | M0          | 0.404481 | 0.207781 | 0.182219 |
| Cosmetics   | M1          | 0.404974 | 0.207691 | 0.172627 |
| Cosmetics   | M2          | 0.409534 | 0.210913 | 0.213567 |
| Cosmetics   | M3          | 0.409709 | 0.210732 | 0.202495 |
| Cosmetics   | M4          | 0.410786 | 0.21027  | 0.219662 |
| Cosmetics   | M5          | 0.40974  | 0.211012 | 0.204988 |
| Coveo       | M0          | 0.505594 | 0.127658 | 0.468111 |
| Coveo       | M1          | 0.509207 | 0.127164 | 0.467374 |
| Coveo       | M2          | 0.522178 | 0.126446 | 0.464535 |
| Coveo       | M3          | 0.523457 | 0.126201 | 0.46658  |
| Coveo       | M4          | 0.524391 | 0.126271 | 0.460127 |
| Coveo       | M5          | 0.524442 | 0.125902 | 0.466359 |
| Electronics | M0          | 0.324059 | 0.133651 | 0.21541  |
| Electronics | M1          | 0.377275 | 0.133415 | 0.220973 |
| Electronics | M2          | 0.303516 | 0.162187 | 0.051031 |
| Electronics | M3          | 0.310316 | 0.165819 | 0.283791 |
| Electronics | M4          | 0.270437 | 0.17664  | 0.102062 |
| Electronics | M5          | 0.249452 | 0.180333 | 0.180021 |

| dataset     | condition   | AP                | Brier             | MCC               |
|:------------|:------------|:------------------|:------------------|:------------------|
| Cosmetics   | M2          | 0.45371 (0.01074) | 0.22643 (0.01288) | 0.22747 (0.01254) |
| Cosmetics   | M3          | 0.45947 (0.01608) | 0.22951 (0.01800) | 0.26988 (0.01328) |
| Cosmetics   | M4          | 0.46002 (0.01701) | 0.21606 (0.00653) | 0.27903 (0.02790) |
| Coveo       | M2          | 0.55185 (0.00289) | 0.16136 (0.00258) | 0.48669 (0.00736) |
| Coveo       | M3          | 0.55104 (0.00540) | 0.16166 (0.01241) | 0.48653 (0.00485) |
| Coveo       | M4          | 0.54990 (0.00610) | 0.16154 (0.01350) | 0.48535 (0.00363) |
| Electronics | M2          | 0.32013 (0.02137) | 0.21211 (0.02574) | 0.21238 (0.10733) |
| Electronics | M3          | 0.33222 (0.00469) | 0.19551 (0.01191) | 0.23310 (0.11895) |
| Electronics | M4          | 0.33097 (0.00577) | 0.19609 (0.01191) | 0.11569 (0.11850) |

所有結果均 exploratory；interval 固定模型／公式／門檻，不含 mining 與 optimization-seed 不確定性。各 seed 共用 user/session resamples，不當成獨立三倍樣本；沒有多重比較校正。未自動展開下一輪。