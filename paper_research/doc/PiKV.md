# 论文研读：PiKV

论文链接：https://arxiv.org/html/2508.06526

该论文讲述了针对MoE架构的KV Cache管理系统。

该文的第一作者来自Yale大学，成功发表于ICML 2026，值得一看。

## 问题背景

在大模型推理中：

- **KV Cache** 用于存储历史 token 的 Key / Value，用于自回归注意力
- **上下文越长，KV Cache 越大**，内存和访存成本急剧上升
- 在 **多 GPU / 多节点** 推理时，KV Cache 还带来严重的通信开销

MoE（Mixture of Experts）虽然，通过 top-k routing **减少计算量**，但是：

- **KV Cache 仍是 dense 的**
- 每个 expert 的 KV 通常被复制或全局同步
- 导致 **内存浪费 + 跨 GPU 查 KV 的高延迟**

该论文要解决的是：

Can we design a KV caching system that is both sparsity-aware and system-optimized for distributed MoE inference?

能否为 MoE 架构设计一个『系统级、稀疏感知、分布式』的 KV Cache 管理方案？

聚焦到现在的MoE系统，面临着以下问题：

- dense KV cache replication
- non-adaptive expert selection
- cache scheduling agnostic to query dynamics

作者将系统瓶颈归纳为3个chanllenge，为了更好地讨论这些问题，我们首先做如下定义并明确一些事实：

- 序列长度 $L$：上下文 token 数（长上下文时 $L$ 很大，比如 32K/64K/128K）。
- 专家数 $E$：MoE 层里可选 expert 的数量。
- 表示维度 $d$：每个 token 在每层产生的 key/value 向量维度（隐藏维度相关）。
- top-k routing：每个 token 只激活 $k \ll E$ 个 experts。

在传统系统实现（Dense Caching）中，**为方便访问与并行，系统会把每个 expert 的 KV 都在每张 GPU 上复制或保持全局一致视图**，如果每个 expert 都维护长度 $L$ 的 KV 序列，每个 token 的 KV 大小量级 $\Theta(d)$，那么每个 expert 的 cache 量级 $\Theta(Ld)$，总共 $E$ 个 expert就占据如下空间：

$$
\text{Mem}_{\text{dense}} \sim E \cdot L \cdot d
$$

可估计Dense Caching 的内存成本是 $O(L\cdot d\cdot E)$（per GPU），**这强调的是“复制/全量保留”的上界**，不是说所有实现都必须这样，但说明了为什么在长上下文+多专家下会爆内存。

**Challenge 1：Expert-sharded KV fragmentation（路由导致的碎片化）**

token-level routing 把不同 token 分配到不同 experts，进一步在多 GPU 情况下把 KV 也分散在不同设备上。

论文用：

$$
KV_t^{(e)} \in \mathbb{R}^{k\times d},\quad e\in \mathcal{R}(q_t)
$$

表达对当前 query $q_t$，只会涉及到路由到的那 $k$ 个 experts 的 KV 子集。

**历史 token 的 KV 不再是一个连续、局部可访问的数组**，而变成按 expert / 按设备分片的多个段，且这些段的时间顺序被打散（时间局部性被破坏）。

MoE 路由把历史 token 的 KV 切散了：

- token A 进 expert 3
- token B 进 expert 11
- token C 又进 expert 3

这样就造成了fragmentation ，从而导致：

- 更多小而散的跨 GPU gather
- 更差的 cache line / page 局部性
- 更难做高效预取与批量通信

**Challenge 2：Latency bottleneck from sparse lookup（稀疏计算≠低延迟）**

论文给了一个期望延迟的分解：

$$
\mathbb{E}[\text{Latency}] \sim O(k\cdot T_{\text{lookup}} + T_{\text{sync}})
$$

并指出 $T_{\text{sync}}$ 来自 inter-GPU 通信。

这不是严格等式，是系统层面的量级分解，表达两类成本：

1. **lookup 成本**：对每个被激活 expert，需要去查它对应的 KV（可能本地，也可能远端）
2. **同步/通信成本**：跨 GPU 拉取 KV 或做 all-to-all，需要同步/等待网络完成

可以看到，MoE虽然减少了计算量，但是 KV 访问从『连续读取一块大内存』变成『跨设备取 k 份分散数据』，通信和等待可能上升。

 **Challenge 3：Non-coordinated Routing, Compression, and Scheduling（策略失配）**

论文指出以往系统把三件事当成互不相关模块：

- expert routing：决定算哪些 expert，只看 gating score，不看 KV 在哪、有没有、贵不贵
- KV compression：决定 KV 如何降维/量化，只看内存/精度，不知道未来会不会用
- cache scheduling：决定哪些 KV 保留/淘汰， 只看历史访问，不知道未来路由会来不来

但存在典型失配场景：

- router 选了某个 expert（因为 gating score 高）
- 但该 expert 的相关历史 KV 由于 scheduling 已被淘汰，或被强压缩到不可用
- 结果 attention recall 变差（精度掉）或需要额外跨设备补救（延迟升）

这三者不协调而产生失配，实际上应该协调这三个模块。

这个三个挑战可以联系成：

1. **MoE routing 导致 KV 分散** 造成fragmentation**（Challenge 1）**
2. fragmentation + 分布式访问 导致lookup + sync 变重 继而latency过高**（Challenge 2）**
3. 为了省内存会做 compression/scheduling，但如果不与 routing 联动则导致miss / 质量下降 / 额外通信**（Challenge 3）**

所以 PiKV 才会强调『统一地协调 routing/compression/scheduling』，把 KV 作为中心抽象。

要深挖这个问题的话，可以继续观察：

- 多少比例的 token 被路由到“cache hit 率低”的 experts
- 命中率下降与 accuracy/latency 的相关性
- scheduling 的 eviction 与后续路由需求之间的冲突率

PiKV针对这三个挑战进行设计：

- Routing is query- and cache-aware，额外考虑：该 expert 的 KV 是否本地/是否 recently evicted/访问是否昂贵，兼顾系统意义上的 locality：**空间局部性**（KV 在不在当前设备）/**时间局部性**（该 expert 的 KV 是否热）/**通信局部性**（是否减少跨 GPU lookup）
- Compression is hierarchical and expert-partitioned，压缩策略是 **expert-local 的**，每个 expert 可选择不同压缩比 / 方法，并且压缩被视为一个**随时间、随 expert、随 token 变化的过程**，而非静态操作。另外压缩不应破坏未来还会被用到的 KV，决策必须依赖 reuse 预测
- Scheduling is jointly optimized with routing，联动routing

将原本互相独立的问题看成一个耦合的问题进行优化，routing / compression / scheduling 这三个系统 knobs 上做联合优化，定义系统目标函数：

```math
\min_{R,C,S} \; \mathbb{E}_{q\sim Q}
\big[
\text{Latency}(q)
+ \lambda_1 \cdot \text{Memory}(q)
- \lambda_2 \cdot \text{Fidelity}(q)
\big]
```

其中：

- **Latency(q)**：KV lookup/通信/同步等待开销

- **Memory(q)**：KV 占用/压缩后的 footprint

- **Fidelity(q)**：attention recall/输出精度的 proxy

**决策空间的定义**：

- $R(q)\subseteq\{1,\dots,E\}$：routing 决策
- $C: KV \to CompressedKV$：压缩映射
- $S: Cache \to EvictableSet$：调度映射

## 系统设计

TODO

## 实验复现

TODO