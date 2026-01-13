# 论文研读：ShadowKV

论文链接：https://arxiv.org/abs/2410.21465

该文章由ByteDance Seed贡献，成功发表于ICML 2025，这个工作看起来非常的出色，本文将用尽可能详实的语言讲解该论文。

## 基本概念

为了讲透这篇论文，我们要先对一些关键概念进行介绍，必要时会进行数学推导。

**稀疏注意力（Sparse Attention）是什么？**

不是对所有历史 token 做 attention，而是只对『一小部分重要 token』计算 attention；核心目标是减少计算量和减少内存访问。

与之相对的是**标准注意力**，即对于一个 query token $q_t$：

$$
\text{Attn}(q_t) = \sum_{i=1}^{S} \text{softmax}(q_t k_i^\top) \cdot v_i
$$

其中：

- $S$ 是上下文长度（比如 128K / 1M）
- 每一步生成，都要访问 **全部 KV**

当 $S$ 很大时，我们就遇到了老生常谈的问题：

- GPU 带宽瓶颈
- 显存 OOM
- batch size 被迫变小

稀疏注意力认为不是每个历史 token 都对当前 token 重要。

核心思路是把 attention 从：

$$
\sum_{i=1}^{S}
$$

变成：

$$
\sum_{i \in \mathcal{I}_t}, \quad |\mathcal{I}_t| \ll S
$$

其中，$I_t$ 是**被选中的 token 子集**，这就是『稀疏性』的来源

那么就引入了两个关键问题：

1. 选哪些 token？
2. 怎么保证不掉精度？

所有稀疏注意力论文，**本质都在回答这两个问题**，这就引出了不同的流派，在此就不做详细探讨了。

**KV缓存驱逐（KV cache eviction）是什么？**

核心思想就是，在 KV cache 放不下全部历史 token 时，永久删除一部分旧 KV，只保留少量关键 KV，该方法趋向于删除『未来不重要的 KV』，只保留**最近的 token/高频被关注的 token/特殊结构 token**，这和稀疏注意力的一大不同点是，该策略会**永久删除**一些KV，导致不可逆的信息丢失。

**什么样的矩阵是低秩结构的？**

对一个矩阵 $X \in \mathbb{R}^{n \times d}$：

- rank($X$) 等于线性无关列（或行）的数量
- 如果：

$$
\text{rank}(X) \ll \min(n, d)
$$

那么 $X$ 是**低秩的**，也就是线性无关列的数量远小于行数和列数。

不妨举例：

- 存了 10 万个 token 的 Key
- 每个 Key 是 128 维

如果这些 Key都是某 20 维子空间里的线性组合，那**根本不需要存 128 维**，存 20 维系数 + 投影矩阵就够了。

**投影矩阵是什么？**

直观理解是，把任意向量投影到某个子空间上的算子，保留我们关心的子空间成分，丢掉其他方向。

具体而言，先用举证来表示一个空间，设 $\mathcal{S}$ 是一个 **r 维子空间**，用一个矩阵：

$$
V = [v_1, v_2, \dots, v_r] \in \mathbb{R}^{d \times r}
$$

其中：

- $v_i$ 是子空间的一组**正交归一基**
- 满足：

$$
V^\top V = I_r
$$

表示出子空间：

$$
\mathcal{S} = \{ V a \mid a \in \mathbb{R}^r \}
$$

可见，子空间就是所有这些向量的集合。

现在给定：

- 一个向量 $x \in \mathbb{R}^d$，这是原始的高维向量
- 一个子空间 $\mathcal{S}$，上面的 $x$ 处于比  $\mathcal{S}$ 更大的空间里

对于一个要被压缩/近似的原始高维向量 $x$，我们采用欧几里德距离衡量信息损失，于是要找：

$$
\hat{x} \in \mathcal{S}
$$

使得：

$$
\|x - \hat{x}\|_2 \;\text{最小}
$$

这是一个标准的最小二乘问题，可以如下推导：

目标函数：

$$
f(a) = \|x - Va\|_2^2
= (x - Va)^\top (x - Va)
$$

对 $a$ 求导：

$$
\nabla_a f = -2 V^\top x + 2 V^\top V a
$$

因为：

$$
V^\top V = I
$$

所以：

$$
\nabla_a f = -2 V^\top x + 2 a
$$

令梯度为 0：

$$
a = V^\top x
$$

接着得到投影的最终形式：

$$
\boxed{
\hat{x} = V V^\top x
}
$$

这说明投影是一个**线性算子**，由矩阵 $P = V V^\top$ 完成（这就是投影矩阵），下面用一个例子表示。

我们先假设一个三维空间：

$$
\mathbb{R}^3
$$

这是**完整空间**，可以理解为：每个向量有 3 个维度。

一个低维子空间（关键假设），现在我们假设：数据其实主要分布在一个 **2 维平面** 上。

定义这个平面为：

$$
\mathcal S = \text{span}\{v_1, v_2\}
$$

其中：

$$
v_1 = 
\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix},
\quad
v_2 =
\begin{bmatrix}
0 \\ 1 \\ 0
\end{bmatrix}
$$

也就是说：

- 子空间 = **xy 平面**
- 第三维（z）是『次要 / 噪声方向』

现在我们取一个具体向量：

$$
x =
\begin{bmatrix}
1 \\
2 \\
0.8
\end{bmatrix}
$$

这个向量**不在子空间中**，我们要对其进行投影。

在所有平面内的点中，找一个 $\hat x$，
使得：

```math
\boxed{
\min_{a_1,a_2}
\left\|
\begin{bmatrix}
1 \\ 2 \\ 0.8
\end{bmatrix}
-
\begin{bmatrix}
a_1 \\ a_2 \\ 0
\end{bmatrix}
\right\|_2^2
}
```

把误差写出来：

```math
\|x-\hat x\|_2^2
=
(1-a_1)^2
+
(2-a_2)^2
+
(0.8-0)^2
```

注意一件事（非常关键）：

- 第三项 $0.8^2$ 和 $a_1,a_2$ 无关
- 所以最小化问题等价于：

$$
\min_{a_1,a_2}
\Big[(1-a_1)^2 + (2-a_2)^2\Big]
$$

这个解可以一眼看出来，就不按部就班算了。所以立即推，得到最后的最小二乘解：

$$
\boxed{
\hat x =
\begin{bmatrix}
1 \\ 2 \\ 0
\end{bmatrix}
}
$$

投影矩阵自然就是：

$$
P = VV^{T} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

**Pre-RoPE具有低秩特性？**

旋转前的Key，不含位置信息，序列间所有 token 共享同一线性投影，所有token都几乎在同一子空间附近（这是『模型 + 表示方式 + 层级 + 位置编码』共同作用下的经验规律）。

那么，我们不难得出一个事实：

设有一组 token 表示：

$$
k_1, k_2, \dots, k_S \in \mathbb{R}^d
$$

如果存在一个 **固定子空间** $\mathcal S \subset \mathbb{R}^d$，

$$
\dim(\mathcal S) = r
$$

使得：

$$
k_i \in \mathcal S,\quad \forall i
$$

那么，把它们叠成矩阵：

$$
K =
\begin{bmatrix}
k_1^\top \\
\vdots \\
k_S^\top
\end{bmatrix}
\in \mathbb{R}^{S \times d}
$$

**必然有：**

$$
rank(K) ≤ r
$$

**Post-RoPE为什么会破坏低秩特性？**

旋转后的Key，每个 token 被施加 **不同的旋转**，同一个向量，在不同位置方向完全不同，即『语义 + 位置信息混合后的 Key』，于是**破坏了低秩特性**，如下可作推导来体现这一特点。

先从**Pre-RoPE 的结构**出发，设：

$$
k_i^{\text{pre}} \approx V a_i
$$

- $V \in \mathbb{R}^{d\times r}$：共享子空间
- $a_i$：token-specific 坐标

于是：

$$
K_{\text{pre}} \approx A V^\top
\quad\Rightarrow\quad
\text{rank}(K_{\text{pre}}) \le r
$$

这是Pre-RoPE的特性。

对第 $i$ 个 token：

$$
k_i^{\text{post}} = R_i\, k_i^{\text{pre}}
$$

其中$R_i$ 是一个 **正交旋转矩阵**，**但依赖于位置 $i$**，接着回代：

$$
k_i^{\text{post}} = R_i V a_i
$$

注意这里的关键变化：

- 原来：

  $$
  k_i^{\text{pre}} \in \text{span}(V)
  $$

- 现在：

  $$
  k_i^{\text{post}} \in \text{span}(R_i V)
  $$

**$R_i V$ 对不同的 $i$，一般不是同一个子空间**，也就是说：

- token 1 用的是子空间 $R_1 V$
- token 2 用的是子空间 $R_2 V$
- token 3 用的是子空间 $R_3 V$

把所有 post-RoPE Key 堆起来：

$$
K_{\text{post}} =
\begin{bmatrix}
(R_1 V a_1)^\top \\
(R_2 V a_2)^\top \\
\vdots
\end{bmatrix}
$$

**无法**写成：

$$
K_{\text{post}} \approx A V^\top
$$

因为每一行的『右侧基』都不一样。

**SVD**： 矩阵分解，设 $A \in \mathbb{R}^{m\times n}$，其奇异值分解（SVD）为：

```math
A = U Σ Vᵀ,  
Σ = diag(σ₁, σ₂, …, σᵣ),  σ₁ ≥ σ₂ ≥ … ≥ σᵣ > 0
```

对任意给定秩 $k$，令

```math
A_k = U Σ_k Vᵀ,

where Σ_k = diag(σ₁, …, σ_k) and all remaining entries are zero.
```

这就是**截断 SVD**，只保留前 $k$ 个奇异值及对应奇异向量，这就有一定程度上削减了信息量。

则在所有秩不超过 $k$ 的矩阵中，$A_k$ 是 **最优近似**，也就是误差最小，并且对以下任意酉不变范数（unitarily invariant norm）都成立：

$$
‖A − A_k‖ = min_{rank(B) ≤ k} ‖A − B‖
$$

利用这个性质，可以**通过低秩近似，用更小的矩阵来表示原始的Key矩阵，从而显著减少存储和计算开销**。

在实际训练好的 Transformer 中，有一些特性：

- Key 矩阵在 **token 维度上高度相关**
- 信息往往集中在少数主方向上

这意味着 Key 矩阵在数值上线性相关性强，**有效秩（effective rank）远小于 $d_k$**。

对某一层、某一 head 的 Key cache：

$$
K \in \mathbb{R}^{T \times d_k}
$$

做奇异值分解：

$$
K = U \Sigma V^\top
$$

其中：

- $U \in \mathbb{R}^{T \times r}$
- $\Sigma \in \mathbb{R}^{r \times r}$
- $V \in \mathbb{R}^{d_k \times r}$
- $r = \text{rank}(K)$

只保留前 $r' \ll d_k$ 个最大奇异值：

$$
K \approx U_{r'} \Sigma_{r'} V_{r'}^\top
$$

这是在 Frobenius 范数意义下的最优低秩近似。

可以利用低秩近似减少存储量，如下是Key Cache的原始存储：

$$
\text{Storage}(K) = T \cdot d_k
$$

SVD 压缩后存储：

$$
\text{Storage}(U_{r'}) + \text{Storage}(\Sigma_{r'}) + \text{Storage}(V_{r'})
= T \cdot r' + r' + d_k \cdot r'
$$

当 $r' \ll d_k$ 且 $T$ 很大时：

$$
T \cdot r' + d_k \cdot r' \ll T \cdot d_k
$$

于是显存占用显著下降。

推理阶段要等价重写，标准注意力中 Query–Key 乘积为：

$$
QK^\top
$$

代入低秩近似后得到：

$$
QK^\top \approx Q (V_{r'} \Sigma_{r'} U_{r'}^\top)^\top
= (Q V_{r'} \Sigma_{r'}) U_{r'}^\top
$$

关键点：

- Query 先被投影到低维子空间（$r'$）
- 再与时间维度上的 $U_{r'}$ 做内积
- 避免了显式重建完整 $K$

可以总结：在 LLM 推理中，通过对注意力 Key 矩阵施加基于奇异值分解的低秩近似，可以将原本随上下文长度线性增长的高维 Key 存储，重参数化为一组低维因子表示。在保持主要注意力子空间信息的前提下，该方法显著降低了 Key cache 的存储复杂度和内存访问成本，从而提升了长上下文推理的可扩展性。

## 背景提要

前置概念已经基本囊括，现在开始正式讲解这篇论文。

我们对KV Cache已经很熟悉了，KV Cache在大模型的长上下文推理过程中可能会爆炸，从而超出显存容量。于是，系统会驱逐一部分缓存，将一部分数据Offload到CPU主存。

但是过去的方法存在着三大问题：

- Accuracy Degradation
- Inadequate Memory Reduction
- Significant Decoding Latency Overhead.

KV Cache驱逐策略会导致信息的丢失和准确度的下降；动态稀疏注意力方法保留了所有的KV对在GPU，选择不全量计算所有的KV，而是有选择地计算部分KV，不过这类方法只是减轻了计算开销，并没有降低内存地使用；如果简单地将KV卸载到CPU的话，又会带来巨大的延迟。

一个理想的系统，应该具备如下特性：

- Reduce GPU Memory Usage
- Minimize Inference Latency
- Maintain Accuracy within Limited Sparse KV Cache Budgets

## 核心方法

本文有几个关键发现，作者基于此设计了更优秀的系统。

pre-RoPE具有低秩特性，**虽然不同序列的低秩子空间相似度很低，但是同序列及其后续序列的子空间有极高相似度，也就是共享同一子空间**，那么也意味着优秀的压缩率。

也就是说，对于一个序列的 pre-RoPE keys：

$$
K = [k_1, k_2, \dots, k_T] \in \mathbb{R}^{T \times d}
$$

若存在：

$$
K \approx U_r V_r^\top,\quad r \ll d
$$

则说明：

- 所有 key 基本落在同一个 **r 维子空间**
- 新 token 的 key 只是该子空间中的**线性组合**

作者对 **LLaMA-3.1-8B** 的以下对象做了 **奇异值分解（SVD）**，并画出了**相对奇异值分布**：

- 权重矩阵：
  - $W_k$：Key 投影权重
  - $W_v$：Value 投影权重
- 输入激活：$X$
- **Key Cache（KV Cache）**
  - RoPE 之前的 Key（pre-RoPE）
  - RoPE 之后的 Key（post-RoPE）
- Value Cache

pre-RoPE 的 Key 是所有对象中秩最低的，也就是说大部分信息集中在**很少几个奇异值**里，后面的奇异值非常小即冗余很大。

基于这个发现，我们可以对pre-RoPE Key做低秩分解，从而存储的更少的数据量，几乎不影响准确度，带来的额外开销在prefill期间也可以忽略，因为低秩分解是线性空间复杂度，而prefill本身是平方级别的。

要补充一点，pre-RoPE只在prefill期间使用，至于具体原因会在下一节讨论。

低秩分解后，成功削减了pre-RoPE Key的显存占用，这时候可以把pre-RoPE Key Offload到CPU主存中。

在存储时，将 Key cache 表示为：

$$
K \;\;\longrightarrow\;\;
\begin{cases}
C = U_r \Sigma_r \in \mathbb{R}^{T \times r} & \text{(token-wise coefficients)} \\
B = V_r \in \mathbb{R}^{d_k \times r} & \text{(shared basis)}
\end{cases}
$$

存储复杂度由：

$$
\mathcal{O}(T \cdot d_k)
\quad\rightarrow\quad
\mathcal{O}(T \cdot r + d_k \cdot r)
$$

当 $r \ll d_k$ 且 $T$ 较大时，内存占用显著降低。

但是Value还是存在GPU上，因为Value是没法做这种低秩分解的，因为它们不符合Key所具有的低秩特性，原因会在后文讨论。

要注意的是，由于进行了低秩分解，所以在计算注意力时，要将进行Recovery，恢复出完整信息，而不能直接在低秩空间中进行。

对任意 token $i$，其 pre-RoPE Key 的低秩表示为：

$$
k_i \approx C_i B^\top
$$

其中：

- $C_i \in \mathbb{R}^{r}$ 为第 $i$ 行系数
- $B^\top \in \mathbb{R}^{r \times d_k}$

恢复操作是把缓存里的 $(C,B)$ 取出来做近似， 该操作定义为：

$$
\hat{k}_i^{\text{pre}} = C_i B^\top \in \mathbb{R}^{d_k}
$$

这是一个线性重构过程，在误差最小化意义下等价于最优低秩近似，所以不可能恢复到原来一样的，对于具体恢复方法会在后面讨论。

另外，恢复过程和访存是可以Overlap的，它们被放到不同的CUDA stream中并行执行，相比传统系统，大大提升了性能。

上面解决的是Offload问题，那么下面要解决的是：在 sparse attention（稀疏注意力）中，如何在 **Top-K 极小（≈1.56%）** 的情况下，
 **既保持注意力精度，又显著降低 decoding latency？**

这里的选择的TOP-K只占全部的1.56%，也呼应了『稀疏』这一特性，所以KV的选择至关重要，如果 KV 选择不准，在这么小的 K 下精度一定会崩。在这一问题下，作者有一个关键发现：对于在时间上相邻的Token，post-RoPE Key的方向非常相近。

也就是在时间维度上，Key 表示是**局部平滑的**：

$$
\cos(k_t, k_{t+1}) \approx 1
$$

这个关键特点很好地支撑了chunk-level approximation：

- 相邻 token 的 Key 非常相似

- 一个 chunk（连续 token 块）内部 Key 变化很小

- 可以用 **一个代表（landmark）** 近似整个 chunk

- KV selection 先选 chunk，再选 token，这样极大地将低了复杂度

不过不能过度乐观，因为：

- **并非所有 chunk 都平滑**
- 有极少数 chunk：
  - 语义突变
  - 结构变化大
  - 近似误差不可接受

这些被称为 **outlier chunks**，占比只有 **0.3%**，它们是不能被近似的，不过考虑到数量极少，可以接受全量存储，用极少的显存换取全局精度稳定性。这个问题值得探讨一下，如果针对这些outlier chunks进一步优化，或许是一个不错的创新点。

可以总结一下文章提出的两个重要发现和结论：

- 同一序列Token的pre-RoPE Key具有低秩特性，因此可以利用SVD低秩近似，以更少空间Offload到CPU
- 时间上相邻的Token的Key高度相似，于是可以使用chunk优化复杂度，加速KV选择

基于第一个发现，作者在Prefill阶段设计了如下流程，对 Transformer 的 **KV cache** 进行结构化压缩与分层存储：

![shadowkv_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/ShadowKV_2.png)

其中，输入张量如下：

- $K \in \mathbb{R}^{b \times h_{kv} \times s \times d}$：**pre-RoPE Key cache**
- $K^{\text{RoPE}} \in \mathbb{R}^{b \times h_{kv} \times s \times d}$：**post-RoPE Key cache**
- $V \in \mathbb{R}^{b \times h_{kv} \times s \times d}$：**Value cache**

超参数：

- $r$：SVD 截断秩
- $c$：chunk size（token 数）
- $o$：outlier chunk 数量

**Step 1：对 pre-RoPE Key 做低秩分解（核心压缩）**

$$
A \in \mathbb{R}^{b \times s \times r}
B \in \mathbb{R}^{b \times h_{kv} \times r \times d}
\leftarrow \text{SVD}(K)
$$

对每个 batch 的 Key 矩阵：

$$
K \approx A B
$$

这就是一个 **rank-$r$** 的近似，其中：

- $A$：token 维度的低维表示（随序列变化）
- $B$：head 共享的投影基（相对稳定）

这个 $AB$ 才是真正要存的Key Cache数据，这也意味着Decoding阶段使用时，必须进行恢复。

**Step 2：对 post-RoPE Key 做 chunk 化与均值聚合**

$$
K^{\text{RoPE}} = \text{RoPE}(K)
$$

$$
C \in \mathbb{R}^{b \times h_{kv} \times (s/c) \times d}
\leftarrow \text{Reduce}(K^{\text{RoPE}})
$$

其中 Reduce 表示：将连续 $c$ 个 token 的 Key 求均值。

动机：

- post-RoPE Key 在**局部上下文**中变化缓慢
- chunk mean 可作为该段 token 的 **landmark 表示**

**Step 3：计算 chunk 内 Key 的相似度**

$$
S \leftarrow \text{CosineSimilarity}(C, K^{\text{RoPE}})
$$

含义：

- 对每个 token，计算其 Key 与所属 chunk 均值的 cosine similarity
- $S$ 衡量 token 是否“服从”该 chunk 的主方向

其中

- 高相似度 → token 是该 chunk 的"典型成员"
- 低相似度 → token 可能对 attention 影响较大（潜在 outlier）

**Step 4：识别 outlier chunks（关键稀疏化步骤）**

$$
I \leftarrow \text{ArgTopK}\big(-\min(S, \text{dim}=-1),\ o\big)
$$

解释：

- 对每个 chunk，取 **最小 cosine similarity**
- 越小则chunk 内存在强偏离 token
- 选择最差的 $o$ 个 chunk 作为 **outliers**

这一步在做一种 **结构化异常检测**：

- 非典型 Key 对 attention 结果影响更大
- 必须保留其精确表示

**Step 5：提取 outlier KV（保留在 GPU）**

$$
K^{\text{outlier}}, V^{\text{outlier}}
\leftarrow \text{Gather}(K^{\text{RoPE}}, V, I)
$$

对这些 outlier chunks**不做压缩**，完整保留在 GPU，用于精确 attention，不offload到CPU内存。

**Step 6：非 outlier 的 Value 下沉 到CPU内存 + 将非离群的Key Chunk 用 landmark 表示**

$$
V^{\text{CPU}} \leftarrow V \setminus V^{\text{outlier}}
$$

$$
L \leftarrow C \setminus \text{Gather}(C,I)
$$

将剩下的Value也就是非离群的Value卸载到CPU内存，Key Chunk用Landmark表示。

接下来讨论第二个关键设计，旨在将低稀疏注意力的延迟开销，不影响准确度情况下，选择部分KV进行重建。

在 **prefill 阶段（Algorithm 1）**，ShadowKV 已经完成了：

- pre-RoPE Key 的低秩分解：$K \approx AB$
- post-RoPE Key 的 chunk landmark 表示：$L$
- 将大部分 Value cache 下沉至 CPU，仅保留 outlier

**Decoding 阶段的核心挑战是：**在 **每一步生成（autoregressive decoding）** 中，如何在 **不恢复完整 KV cache 到 GPU** 的情况下，近似原始 attention 结果。

作者提出了如下算法：

![shadowkv_3](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/ShadowKV_3.png)

其中，输入张量：

- $A \in \mathbb{R}^{b \times s \times r}$，$B \in \mathbb{R}^{b \times h_{kv} \times r \times d}$：pre-RoPE Key 的低秩分解
- $L \in \mathbb{R}^{b \times h_{kv} \times n_c \times d}$：非 outlier chunks 的 **post-RoPE Key landmarks**
- $V^{\text{CPU}}$：下沉至 CPU 的 Value cache
- $K^{\text{outlier}}, V^{\text{outlier}}$：保留在 GPU 的 outlier KV
- $Q \in \mathbb{R}^{b \times h_q \times s_q \times d}$：当前解码步的 Query
- $n_c$：chunk 数
- $k$：每个 KV head 允许访问的 chunk 数（budget）

**Step 1：Chunk-level Attention Scoring（粗粒度注意力）**

计算Query–Landmark 点积：

$$
P \in \mathbb{R}^{b \times h_q \times s_q \times n_c}
\leftarrow Q L^\top
$$

这是一个 **Query 对 chunk 均值 Key 的 attention score**，衡量当前 Query 对每个 chunk 的整体相关性。

计算Softmax 归一化：

$$
S \leftarrow \text{Softmax}(P / \sqrt{d})
$$

这是标准 scaled dot-product attention。

沿 token 维度聚合

$$
S_1 \in \mathbb{R}^{b \times h_q \times n_c}
\leftarrow \sum\nolimits_{s_q} S
$$

聚合所有 query token 的注意力得分，得到 **chunk 级别的重要性评分**。

Query head → KV head 映射：

$$
S_2 \in \mathbb{R}^{b \times h_{kv} \times n_c}
\leftarrow \text{max}_{kv\_group}(S_1)
$$

在 GQA / MQA 中，多个 query head 共享一个 KV head，用 max 聚合可保留最"激进"的关注信号。

**Step 2：Top-k Chunk Selection（稀疏化）**
$$
I \in \mathbb{R}^{b \times h_{kv} \times k}
\leftarrow \text{ArgTopK}(S_2, k)
$$

这一步完成了 **chunk-level routing**：

- 每个 KV head 只访问 $k$ 个 chunk
- Attention 从 $O(s)$ 降到 $O(kc)$

这是 ShadowKV **解码加速和显存节省的核心**。

**Step 3：Value Cache 的稀疏恢复**

从 CPU 拉取 Value：

$$
V^{\text{sparse}} \leftarrow \text{Gather}(V^{\text{CPU}}, I)
$$

- 仅恢复被选中 chunks 的 Value
- 避免全量 CPU → GPU 拷贝

构造最终 Value cache：

$$
V \leftarrow [V^{\text{outlier}};\ V^{\text{sparse}};\ V]
$$

包括：

- 全精度 outlier

- 被选中 chunk 的 Value

- 历史 decode token 的 Value

**Step 4：Key Cache 的低秩重建**

从低秩因子恢复 Key：

$$
K^{\text{sparse}}
\leftarrow \text{MatMul}(\text{Gather}(A, I), B)
$$

这是：

$$
K_i \approx A_i B
$$

仅对 **被选中 chunks** 进行恢复。

应用 RoPE 并合并：

$$
K \leftarrow [K^{\text{outlier}};\ \text{RoPE}(K^{\text{sparse}});\ K]
$$


RoPE 在重建后再应用，保证位置编码与原模型一致。

至此，我们已经讲完了整套方法，大致流程如下图所示：

![shadowkv_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/ShadowKV_1.png)

##  补充讨论

**为什么在 decoding 阶段只能使用 post-RoPE Key，而不能使用 pre-RoPE Key？**

在带 **RoPE（Rotary Positional Embedding）** 的 Transformer 中，解码阶段的注意力计算为：

```math
\mathrm{Attn}(Q, K, V)
=
\mathrm{Softmax}
\left(
\frac{
(\mathrm{RoPE}(Q))
(\mathrm{RoPE}(K))^\top
}{\sqrt{d}}
\right)
V
```

本质问题是：**在 decoding 阶段，是否可以用 pre-RoPE 的 Key 参与 attention？**

根据公式，这显然不行，因为Query必然是post-RoPE，Key要和Query点积，两者必须处于同一 RoPE 坐标系。

**为什么Key具备低秩特性，而Value不具备低秩特性？**

在注意力机制中，Key 仅作为判别性路由信号，其有效自由度由 Query 子空间限制，因此自然呈现低秩结构；而 Value 直接决定输出表示，其信息熵与表达自由度必须保持较高水平，因而不具备可利用的低秩特性。

**如何恢复被低秩近似的Key Cache？**

设原始 **pre-RoPE** Key cache（按单个 batch、单个 KV head 展开 token 维）为：

$$
K \in \mathbb{R}^{s \times d}
$$

对它做截断 SVD（或任意等价的低秩分解）得到 rank-$r$ 近似：

$$
K \approx K_r = U_r \Sigma_r V_r^\top
$$

常见存储方式有两类（都能恢复）：

**形式 A：**

$$
A := U_r\Sigma_r \in \mathbb{R}^{s \times r},\quad
B := V_r^\top \in \mathbb{R}^{r \times d}
$$

则

$$
\hat K = AB \in \mathbb{R}^{s \times d}
$$

**形式 B：**

$$
\hat K = U_r \Sigma_r V_r^\top
$$

先在 $A$ 上取行（gather）：

$$
A_I \in \mathbb{R}^{m \times r} \leftarrow \text{GatherRows}(A, I)
$$

再乘回基 $B$：

$$
\boxed{\hat K_I = A_I B \in \mathbb{R}^{m \times d}}
$$

这一步是 ShadowKV decoding 的关键：**只为被路由选中的 chunks/token 重建 Key**，避免 $s\times d$ 的全量开销。

## 实验复现

基于Docker镜像进行复现，首先拉取镜像：

```powershell
docker pull nvidia/cuda:12.1.1-devel-ubuntu22.04
```

启动容器：

```powershell
docker run -it --rm \
  --gpus all \
  -v `pwd`/ShadowKV:/ShadowKV \
  nvidia/cuda:12.1.1-devel-ubuntu22.04 \
  bash
```

安装工具链：

```powershell
apt update
apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    build-essential \
    git \
    ca-certificates
```

升级pip：

```powershell
pip install --upgrade pip setuptools wheel
```

接下来进入项目目录，按照README执行就行

(Pending)

