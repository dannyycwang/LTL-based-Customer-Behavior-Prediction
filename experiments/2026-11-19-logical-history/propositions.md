# Proposition 與有限軌跡語義

沿用各 dataset 的既有 proposition vocabulary；不是本輪重新選取 propositions。抽象由 `pilot.py::abstract` 按時間順序計算，再取與該 dataset vocabulary 的交集。購買事件代碼 `p` 僅用於標籤稽核；eligible prefix 中沒有購買事件。

| Proposition | 定義 |
|---|---|
| b | 一般頁面瀏覽（Coveo page view；Cosmetics 原事件沒有此 action） |
| d | 商品詳情／view |
| a | 加入購物車 |
| r | 從購物車移除 |
| hp / np | 目前事件有／沒有有效 product ID |
| fp / rp | 目前有效 product 在本次抽象範圍第一次／再次出現 |
| sp / pc | 目前 product 與先前最近有效 product 相同／不同 |
| p2 / p3 | 目前有效 product 在本次抽象範圍已累計出現至少兩／三次（含當前事件） |
| dp1 / dp2 / dp3 | 有效 product 事件發生時，範圍內已出現的不同有效 products 數為一／二／至少三 |
| hc | 目前事件有有效 category ID |
| sc / cc | 目前 category 與先前最近有效 category 相同／不同 |
| rc | 目前有效 category 在本次抽象範圍先前已出現 |

Coveo vocabulary：`b,d,a,r,fp,rp,sp,pc,p2,p3,sc,cc,rc`。Cosmetics 額外包含 `hp,np,dp1,dp2,dp3,hc`。

有效 ID 排除 `P0`／`C0`、`UNK`、空字串、`nan`、`None`。只有有效 ID 更新最近有效 product/category 狀態。Coveo 只在目前 ID 有效時產生相同／不同 flags；Cosmetics 沿用原轉換器，在已有先前有效 ID 時也比較目前缺失 ID。這些資料集既有差異保留並通過原 prefix abstraction replay。

B 從最近十個事件的起點重新初始化所有 product 次數、distinct products、category 集合及前一有效 ID；A 的摘要與 C/D 從 prefix 起點初始化。故相同尾端事件在 B 與 C/D 可有不同關係 propositions。t=10 時起點相同，輸入與所有 candidate satisfaction 均逐列驗證一致。

公式在有限軌跡上評估：`X` 是 strong next，最後位置為 false；`U` 是 strong until，右式必須在剩餘軌跡某位置成立；`F/G` 包含當前位置。保存的 DECLARE response 模板是 `G(a -> F b)`，因此允許同一事件的 b 滿足 a 的 response；precedence 模板要求嚴格較早的 a，chain response 為 `G(a -> X b)`。採用的是既有模板的具體語義，不假設所有 DECLARE 套件具有完全相同的端點／同時事件定義。
