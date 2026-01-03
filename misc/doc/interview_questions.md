# AI大模型面试题
## 20260103

1. 请详细解释一下 Transformer 模型中的自注意力机制是如何工作的？

   - 自注意力在做什么？给定一段长度为 $L$ 的序列表示 $X\in \mathbb{R}^{L\times d_{model}}$ (每个 token 一个向量)，自注意力的核心是：**每个位置 i 都去看序列里所有位置 j，并按相关性加权汇聚信息**。

   - 通过线性层把输入 $X$ 映射成：

     - Query: $Q = XW_Q$
     - Key: $K = XW_K$
     - Value: $V = XW_V$

     其中 $W_Q,W_K,W_V \in \mathbb{R}^{d_{model}\times d_k}$ 

   - 计算注意力打分矩阵：
    ```math
    S=\frac{QK^\top}{\sqrt{d_k}}\quad\in\mathbb{R}^{L\times L}
    ```
   - 对每一行做 softmax：

    ```math
    A = softmax(S) \quad (行和为 1)
    ```
   - 输出：

    ```math
    Y = AV
    ```
   ​	其中， $\sqrt{d_k}$ 是为了让点积的尺度更稳定（否则 $d_k$ 大时 logits 方差变大，softmax 易饱和）。

   - 应用Mask（尤其是自回归模型）

     - **Padding mask**：不让 pad 位置参与
     - **Causal mask**：生成式模型里位置 i 只能看 $\le i$ 的位置，避免"偷看未来"

2. 什么是位置编码？为什么在 Transformer 必需？

   - 自注意力本身对 token 的排列是 **置换不变（permutation equivariant）** 的：
      如果把输入序列整体打乱顺序，同样的 Q/K/V 计算与点积结构并不会"天然"知道谁在前谁在后（它只看向量内容）。
      所以如果不注入位置信息，模型无法区分：

     - 狗咬人 vs 人咬狗

     - A 在 B 前面 vs A 在 B 后面

     因此必须把『位置』作为额外信号加入。