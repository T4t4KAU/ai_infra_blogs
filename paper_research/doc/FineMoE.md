# 论文研读：FineMoE

论文链接：https://arxiv.org/abs/2502.05370

在此之前，我们已经分享过在稠密模型中对KV Cache进行Offload的文章，本篇文章聚焦于MoE，讨论了Expert Offload技术。

该文的第一作者来自美国的史蒂文斯理工学院，发表于EUROSYS'26，是系统领域的顶级会议。

## 前置内容

本节先简单介绍一下MoE，看看MoE的训练和推理，以及和稠密模型的区别。

在 Dense Transformer（也就是我们说的稠密模型） 中，**每一层的所有参数**都会参与**每一个 token** 的前向与反向计算，计算复杂度与参数规模线性相关。

数学上（以 FFN 为例）：

$$
y = W_2 \sigma(W_1 x)
$$

MoE 的核心思想是：**用多个专家网络（Experts）替代单个 FFN，每个 token 只激活其中少数几个专家**

形式上：

$$
y = \sum_{i \in \mathcal{E}(x)} g_i(x) \cdot f_i(x)
$$

其中：

- $f_i$：第 $i$ 个 expert（通常是 FFN）
- $g(x)$：门控网络（Router / Gating Network）
- $\mathcal{E}(x)$：被选中的 Top-k experts（通常 k=1 或 2）

在现代 LLM 中，MoE **几乎只替换 FFN 层**，Attention 仍是 Dense：

```
[Attention] → [MoE FFN] → [Attention] → [MoE FFN]
```

原因：

- Attention 具有全局信息整合属性
- FFN 占据 Transformer 中 **大部分参数量与 FLOPs**

MoE 并非 Dense 的完全替代，而是在算力受限、参数可扩展的前提下，提供更高容量的模型结构。

代价是：

- 训练成本高、系统复杂
- 推理性价比取决于路由与通信优化
- 更适合 **超大规模预训练**

现在看看MoE 的训练机制。

我们要建立路由（Routing / Gating）机制，对于每个 token 表征 $x$：

$$
g(x) = \text{softmax}(W_g x)
$$

选取 Top-k：

$$
\mathcal{E}(x) = \text{TopK}(g(x))
$$


- **Switch Transformer**：Top-1
- **GShard / DeepSeek-MoE**：Top-2

设置专家并行（Expert Parallelism）：

- 不同 experts 分布在不同 GPU / 节点
- token 被 **动态分发（dispatch）** 到对应专家
- 专家计算完成后 **再聚合（combine）**

还要做负载均衡（Load Balancing），若不加约束，路由器可能**塌缩**（大多数 token 进入少数专家），因此引入 **辅助损失（Auxiliary Loss）**。

典型形式（Switch Transformer）：
$$
\mathcal{L}_{aux} = \alpha \cdot N \sum_i p_i \cdot f_i
$$

- $p_i$：路由概率
- $f_i$：实际 token 占比
- 鼓励『**均匀使用专家**』

在反向传播中，只有被激活的Expert才参与梯度更新（单 step 中的**有效参数更新密度极低**，但长期来看每个 expert 都能被充分训练），Router也参与反向传播。

再看看推理。

对每个 token：

1. 计算 router
2. 选择 Top-k experts
3. 只执行被选中的 experts
4. 按 gating 权重加权求和

代入到Transformer中看看整体流程。

首先是Prefill阶段，设输入序列长度 $T$，隐层维度 $d$，、

**Step 1：Embedding + Position Encoding**
$$
X \in \mathbb{R}^{T \times d}
$$
**Step 2：Attention 子层（Dense，所有 token）**

1. 计算：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

2. Self-Attention：

$$
A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- **所有 token 都参与**
- **无 MoE、无路由**

**Step 3：Residual + LayerNorm**
$$
H = \text{LN}(X + A)
$$
**Step 4：MoE-FFN 子层（这里才发生路由）**

对 **每一个 token 向量** $h_t$：

1. Router：

$$
g_t = \text{softmax}(W_r h_t)
$$

2. 选择 Top-k experts：

$$
\mathcal{E}(h_t)
$$

3. 仅在被选中的 experts 上执行 FFN：

$$
f_i(h_t) = W_{2,i}\sigma(W_{1,i}h_t)
$$

4. 加权合并：

$$
\text{MoE}(h_t) = \sum_{i\in\mathcal{E}(h_t)} g_{t,i} f_i(h_t)
$$

**Step 5：Residual**

$$
Y = H + \text{MoE}(H)
$$

**Step 6：进入下一层**

- 下一层是新的 block
- 上一层已经完成

再看看推理阶段是怎么做的，单个 token 在一层里的完整流程。

以下是 **第 $l$ 层，在 decoding step $t$** 的真实执行顺序：

**Step 1：LayerNorm**

$$
h^{(l)}_t \rightarrow \tilde{h}^{(l)}_t
$$

**Step 2：Attention（Dense，使用 KV Cache）**

计算 QKV（只算 Q 是新的）：

$$
\begin{aligned}
Q^{(l)}_t &= \tilde{h}^{(l)}_t W_Q^{(l)} \\
K^{(l)}_t &= \tilde{h}^{(l)}_t W_K^{(l)} \\
V^{(l)}_t &= \tilde{h}^{(l)}_t W_V^{(l)}
\end{aligned}
$$

写入 KV cache（**每层独立**），完成Attention 计算（Query × 历史 KV）

$$
\text{Attn}^{(l)}_t =
\text{softmax}\left(
Q_t K_{1:t}^\top
\right)V_{1:t}
$$

**Step 3：Residual**

$$
h'^{(l)}_t = h^{(l)}_t + \text{Attn}^{(l)}_t
$$

**Step 4：LayerNorm**

$$
\hat{h}^{(l)}_t = \text{LN}(h'^{(l)}_t)
$$

**Step 5：MoE Router（token-level）**

Router 计算：

$$
g^{(l)}_t = \text{softmax}(W_r^{(l)} \hat{h}^{(l)}_t)
$$

选 Top-K experts：

$$
\mathcal{E}^{(l)}_t = \text{TopK}(g^{(l)}_t)
$$

**Step 6：Expert Dispatch**

token embedding 被发送到：

- expert $e_1$
- expert $e_2$（若 Top-2）

**Step 7：Expert FFN（稀疏执行）**

对每个被选 expert：

$$
f_{e}(\hat{h}) = W_{2,e}\,\sigma(W_{1,e}\hat{h})
$$

**Step 8：Combine（加权求和）**

$$
\text{MoE}^{(l)}_t
= \sum_{e \in \mathcal{E}_t}
g^{(l)}_{t,e}\,f_e(\hat{h}^{(l)}_t)
$$

**Step 9：Residual**

$$
h^{(l+1)}_t = h'^{(l)}_t + \text{MoE}^{(l)}_t
$$

随后进入下一层。

流程大致如下：

![FineMoE_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/FineMoE_1.png)

然而，MoE的稀疏性并不代表对系统友好。

每个 expert 通常绑定一个 GPU 或一个 GPU 上的 shard，Token如果负载不均衡，总是被路由到一小部分Expert，就会导致其他 GPU 空闲系统在等待"最忙 expert"。

GPU 擅长的是：大规模、规则、可提前编排的数据并行计算；而 MoE 在推理时产生的是：细粒度、动态、不可预测的条件计算，两者在执行模型上是天然冲突的。

## 问题背景

上文提过，Mixture-of-Experts (MoE) 模型的核心特征是：

- **参数规模极大**（如 40B+）
- **每次推理仅激活少量专家（Top-K）**
- 训练与推理 FLOPs 显著下降

但在 **推理阶段（serving）** 存在严重问题：**尽管大多数专家在当前 token 上不被激活，它们仍必须常驻 GPU 显存**。

论文以三个主流 MoE 模型为例指出：

|     模型     | 非激活参数比例 |
| :----------: | :------------: |
| Mixtral-8×7B |      72%       |
| Qwen1.5-MoE  |      81%       |
| Phi-3.5-MoE  |      84%       |

这表明，GPU 显存浪费严重，吞吐率受限。

于是，过去有一些方法用于优化这一状况，可分为两种：

- Lossy Serving：使用了压缩、裁减和量化技术减小内存需求
- Lossless Serving：将模型的权重进行Offload

## 核心方法

TODO

## 实验复现

TODO
