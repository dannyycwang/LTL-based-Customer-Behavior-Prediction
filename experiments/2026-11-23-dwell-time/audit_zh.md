# 時間可觀察性稽核

兩資料集使用原始 event timestamp，沒有已知可獨立驗證的「系統記錄注意力停留」欄。此次 dwell 一律指 outgoing event gap。incoming gap 對齊目前事件；本輪 outgoing gap 對齊前一個已完成事件，兩者不混用。

程式 `prepare.py::gaps` 先切 timestamps[:t]，再 diff；末端保持 NaN。數值特徵用0作運算佔位但另有 missing mask，bucket 使用 du，故真零秒仍是可觀察的 dl。缺失 endpoint 不被視為短停留；負間隔 fail。原始 timestamp 缺失若使事件排序不確定則 fail，不能悄悄換 cohort。秒單位由毫秒除1000，Cosmetics UTC 原始字串先轉毫秒。ties 用原 row order。

實際回放通過：

| Dataset | Split | Sessions | Known gaps | True zero gaps | Unknown terminal gaps |
|---|---|---:|---:|---:|---:|
| Coveo | train | 29474 | 560006 | 132802 | 29474 |
| Coveo | validation | 6305 | 119795 | 28531 | 6305 |
| Coveo | test | 6328 | 120232 | 28367 | 6328 |
| Cosmetics | train | 7326 | 139194 | 10856 | 7326 |
| Cosmetics | validation | 1046 | 19874 | 1621 | 1046 |
| Cosmetics | test | 2093 | 39767 | 3286 | 2093 |

全部保留 t20 prefixes 的 first20 原始 action 與既有抽象逐位一致。對全部 session 改動第20事件之後的 timestamps，gaps 完全不變；單元檢查另測完整 T 與 dwell propositions 不變、terminal／zero 區分、missing、負間隔和秒轉換。原始 events 對齊與 slice-first API 阻止隱藏 purchase timestamp 進入特徵。此次 test timestamps 只用於無標籤前處理／可觀察性檢查，不做模型排名或參數擬合。

bucket 配方沿用 train-only log1p tertiles after p99 cap，但重新限本輪 t20 training 已知間隔擬合。Coveo 原始 buckets 的 train split 與後來 protected split 未必相同；Cosmetics 舊末端填0與跨 purchase 歷史不能直接搬用；Electronics 舊修補明確用 master CSV，所以本輪不重用其 thresholds。新條件全部共用同一組 bucket parameters，沒有 test cutpoint search。所有 raw vocab、numeric scaler 也只 fit train。

T 維度 Coveo298、Cosmetics394，完整欄名、桶切點、source/cohort/feature hashes 在各 dataset/preparation.json。known gap summary 的空集合以0佔位，另存 known count/all_missing；frequencies denominator 固定 prefix長度或 t−1，沒有無說明的 missing-as-short。

既有 Act+Rel mining input SHA、candidate SHA、全部 training prefix ID/label/order/trace 已核對。其 t10/20/30 舊採樣記錄只用來使兩 vocabulary 的 learning traces 完全匹配；dwell 對各自 prefix 重新切片，沒有額外 landmark 模型。M0/M1 重用前另驗同 cohort/labels/raw/S、encoder/scaler 與求解器；M3 在 M2 重新選。模型與選擇尚未完成時，不宣稱結果支持任一研究假說。
