# 论文研读：Speculative Speculative Decoding

本文是FlashAttention的作者Dao参与的文章，一作是斯坦福的大佬，该论文拿到了ICLR Poster；值得一提的是，ICLR作为机器学习三大会，最近刚刚晋升为了CCF-A（之前是没有评级的），实至名归。

链接：http://arxiv.org/abs/2603.03251v1

本文的讲解尽量减少了数学原理和技术细节，力求最快理解论文核心思想。

## Speculative Decoding

众所周知，一个LLM的自回归解码属于Token-By-Token Decoding，一轮Forward只能生成一个token，效率被认为太低。

那能不能加速这个Decode过程呢？

有人提出了一个关键发现：

> some inference steps are "harder" and some are "easier".

例如给定下面的句子：

```
The capital of France is ___
```

我们的Decoding就是预测下一个词，几乎所有模型都会预测出：Paris，这就是一个Easy Step，小参数的模型也能预测正确，犯不着用上大模型一次昂贵的Forward，我用Qwen2-1.5B实验了一下，确实如此。如果这样简单的步骤让70B模型来做，就非常缺乏性价比了。

所谓的Easy就是大参数模型和小参数模型给出的概率分布相似，不是我们人类觉得因为这是一个常识所以Easy，这点需要澄清。

但是下面这个句子：

```
The movie was ___
```

这时，预测下一个Token时，大小模型的概率分布差异就大了，在Qwen2-1.5B上，预测结果是"released"，而参数大一点的模型预测的是 "breathtaking"，这差别就大了，这是一个Hard Step，小模型无法处理。

小模型的好处就是快但是不适合做Hard Step，大模型适合做Hard Step但是速度太慢。

我们能否知道当前Step是Easy还是Hard的呢，得以合理分配呢？很遗憾，这是做不到的，所以只能去赌一赌，就有了Speculative Decoding，即投机解码/推测采样。

2022年发表的投机解码(Speculative Decoding)已经成为了大模型推理系统中一个重要的技术，下面简单介绍一下这个算法的流程。

假设：

- **p**：target model（大模型）分布
- **q**：draft model（小模型）分布

步骤：

1. draft model 连续生成 $k$ 个 token：

   ```math
   y_1, y_2, ..., y_k
   ```

2. target model 一次性计算这些位置的概率 $p(y_i)$

3. 对每个 token 依次做 **accept/reject**：

接受概率：

```math
\alpha_i = \min\left(1, \frac{p(y_i)}{q(y_i)}\right)
```

如果 token 被拒绝会发生什么？假设第 $i$ 个 token $y_i$ 被拒绝。

此时不会继续使用 draft token，而是从**修正后的分布**采样一个新 token：

```math
r(x) \propto \max(0, p(x) - q(x))
```

更完整写法：

```math
r(x) = \frac{\max(0, p(x) - q(x))}{Z}
```

其中：

```math
Z = \sum_x \max(0, p(x) - q(x))
```

关键点：重新采样的 token **不会再被拒绝**，这是 speculative decoding 的一个重要性质。因此：**从 residual 分布采样的 token 是直接接受的。**这基于一个很简单的数学原理，这里就不作赘述了。

## Overview

现在SD的一个明显的问题是Draft Model要等待Target Model验证结果返回后，才能进行下一轮预测，也就是Draft和Verify是串行执行的，资源利用率低。

SSD的目标就是消除 speculation（draft）和 verification（verify）之间的顺序依赖，核心思想就是：让 draft（草稿生成）和 verify（目标模型验证）并行执行，并提前为可能的验证结果准备下一轮 token。

### Workload

#### I. 初始化阶段

系统启动两个模块：

- **Speculator（Draft Model）**
- **Verifier（Target Model）**

二者在 **不同设备上并行运行**。

首先：

1. target model 对 prompt 进行 **prefill**
2. draft model 也对 prompt 进行 **prefill**

完成 KV cache 初始化。

#### II. 第一轮 speculation

Draft model：

生成一段候选 token 序列，例如：

```
t1 t2 t3 t4
```

这叫 **speculation tokens**，然后把这些 token 发送给 **Verifier**

#### III Target model 验证 speculation

Target model 对 speculation 进行验证：

它会：

1. 顺序检查 draft token 是否符合 target distribution
2. 接受一部分 token
3. 如果出现不匹配，就停止接受
4. 从 residual distribution 采样一个 **bonus token**

因此 verifier 返回：

```
verify outcome = (接受的token数 k, bonus token)
```

例如：

```
接受 t1 t2
bonus token = t5
```

然后把这个 **verify outcome** 发回 draft

#### IV. Draft 预测验证结果并提前生成

在 verifier 进行验证的同时，draft 不会等待

Draft 会：**预测 verifier 可能返回的 outcome**

例如：

```
可能接受 0 个
可能接受 1 个
可能接受 2 个
...
```

对这些可能结果 **提前生成下一轮 speculation**

例如：

```
outcome1 → 下一段token
outcome2 → 下一段token
```

把这些结果存入 **speculation cache**

#### V. 根据验证结果继续生成

当 verifier 返回实际 outcome 时有多种情况。

CASE 1：Cache hit（预测正确）

draft 直接取出提前生成的 token：cache[verify outcome]，立刻进入下一轮生成，无需重新计算。

CASE 2：Cache miss（预测失败）

如果预测错误：draft 使用 **fallback strategy**：

- 重新生成 speculation
- 可以使用 primary draft 或 backup draft

然后继续流程。

### CASES

本节讨论一些情况，更好地梳理这个算法。

#### CASE 1: All Accpeted

- 当前序列前缀是Prefix

- Draft Model 预测了 4 个 token： T1 T2 T3 T4

- 将4个token发送给Target Model验证
- Draft Model 不等 验证结果，构建投机缓存
  - Key 1：4 tokens Accpeted，bonus=T5	Value：Next Step
  - Key 2：4 tokens Accpeted，bonus=T6        Value：Next Step
  - Key 3：3 tokens Accpeted，bonus=T7        Value：Next Step
- Draft Model 等待验证结果返回
- 4个token 全部Accpeted
  - 如果bonus=T5，则命中Key 1，执行Value中的Next Step，直接给出下一轮 draft token 序列
  - 如果bonus=T6，则命中Key 2，执行Value中的Next Step，直接给出下一轮 draft token 序列
  - 如果bonus=T8，Cache Miss，执行Fallback：以Prefix+T1+T2+T3+T4+T8为前缀进行投机预测

#### CASE 2: Not All Accpeted

- 当前序列前缀是Prefix

- Draft Model 预测了 4 个 token： T1 T2 T3 T4

- 将4个token发送给Target Model验证
- Draft Model 不等 验证结果，构建投机缓存
  - Key 1：4 tokens Accpeted，bonus=T5	Value：Next Step
  - Key 2：4 tokens Accpeted，bonus=T6        Value：Next Step
  - Key 3：3 tokens Accpeted，bonus=T7        Value：Next Step
- Draft Model 等待验证结果返回
- 3个token 全部Accpeted，T4 Rejected
  - 如果bonus=T7，命中Key 3，直接执行操作，直接给出下一轮 draft token 序列
  - 如果bonus=T8，Cache Miss，进入Fallback：以Prefix+T1+T2+T3+T8为前缀进行投机预测

#### CASE 3: All Rejected

- 当前序列前缀是Prefix

- Draft Model 预测了 4 个 token： T1 T2 T3 T4

- 将4个token发送给Target Model验证
- Draft Model 不等 验证结果，构建投机缓存
  - Key 1：4 tokens Accpeted，bonus=T5	Value：Next Step
  - Key 2：4 tokens Accpeted，bonus=T6        Value：Next Step
  - Key 3：3 tokens Accpeted，bonus=T7        Value：Next Step
  - Key 4：0 tokens Accpeted，bonus=T8       Value：Next Step
- Draft Model 等待验证结果返回
- 4个token 全部Rejected
  - 如果bonus=T8，命中Key 4，直接给出下一轮 draft token 序列
  - 如果bonus=T9，Cache Miss，进入Fallback：以Prefix+T9为前缀进行投机预测

### Chanllenges

不难发现，这个系统面临着一些挑战

#### Chanllenge 1

Draft model 生成一段候选 token 序列，例如：

```
t1 t2 t3 t4
```

这叫 **speculation tokens**。

然后把这些 token 发送给 **Verifier**。

在 SD 中，target model 会返回一个 **verification outcome**，包括：

1. **接受了多少个 draft token**
2. **bonus token（补充token）是什么**

例如：

```
draft:  t1 t2 t3 t4
verify: 接受 t1 t2
        bonus token = t5
```

SSD 要在 **verify 还没结束前**就开始准备下一轮 speculation，因此必须提前预测：

```
可能 outcome：
(k = 接受token数, t* = bonus token)
```

问题是：

- token 数量可能是 **0~K**
- bonus token 来自 **整个 vocabulary**

因此可能 outcome 数量大约是 (K + 1) × Vocab，这个空间太过巨大，如何在巨大 outcome 空间中预测最可能的验证结果。

#### Chanllenge 2

在 SSD / Saguaro 里，cache hit 的前提是：

verifier 最终给出的 verification outcome $(k, t^*)$，恰好落在 draft 预先准备好的 cache 里。

这里最难预测的其实不是 $k$，而是 bonus token，论文明确说：大多数时候，bonus token 不是直接从 target 分布采样，而是从 **residual distribution** 采样，residual distribution 在温度高时尤其难预测。

那么如何 预测 bonus token 是一个棘手的问题。

#### Chanllenge 3

即使预测得很好，仍然会有 cache miss，尤其在**batch size 大**和**temperature 高**的时候更容易发生。

一旦 miss，就需要 fallback speculation，如果 fallback 太慢会导致整个batch停顿，如果 fallback 太简单，就会生成容易被拒的token，所以要做权衡设计，如何设计高效的 fallback 策略来处理 cache miss 是一个问题。

### System Design & 理论推导

这个SSD系统称为Saguaro，在论文的实际代码中，Target Model被放到4张H100集群上做TP，Draft Model独占了一张H100，于是跑通整个实验要求5张H100...作者在 Alpaca/GSM8k/UltraFeedback/HumanEval 四个数据集上完成了测试。

#### Predicting Verification Outcomes: Building the Saguaro Cache

如上文所说，系统必须构建一个Cache包含了最可能的验证结果，难点在于结果空间太大。

我们要合理设定 Cache 的大小，如果太大的话，Draft Model就不能在Target Model验证完成前构建完 Cache 了，这反而就导致了Target Model 等待 Draft Model，适得其反了。模型太小的话，缓存命中率降低，这自然不是一件好事。

论文定义：

```math
F_k^p := \{v^T = (k', t^*) \in S^T \mid k' = k\}
```

$F_k^p$：当上一轮是由 **primary speculator** 生成时，在位置 $k$ 处放进 cache 的备选验证结果数量

$F_k^b$：当上一轮是由 **backup speculator** 生成时，对应的 fan-out

可以把它简单理解为：

- 在第 0 个未来位置放多少个候选
- 在第 1 个未来位置放多少个候选
- …
- 在第 $K$ 个未来位置放多少个候选

形成一个序列：

```math
\{F_0, F_1, \dots, F_K\}
```

这里 $K$ 是 speculative lookahead 的最远长度。

设定一个约束：

```math
\sum_{k=0}^{K} F_k \le B
```
其中 B = speculation cache 预算

在**总预算**固定的情况下，怎么分配各位置的 fan-out 最合理？

现在要选择 $F_k$ 最大化 cache hit rate，论文发现 cache miss rate ≈ power-law：

```math
1 - p_{hit}(F) = \frac{1}{F^r}
```

等价于：

```math
p_{\text{hit},*}(F) = 1 - \frac{1}{F^r}
```

其中：

- $F$ 是某个位置分配的 fan-out
- $p_{\text{hit},*}(F)$ 是这个 fan-out 下 cache 命中的概率
- $r > 0$ 是幂律指数
- $*$ 表示 p 或 b（primary / backup）

画出这个函数曲线或者对这个函数的F求导，会发现函数的增速随着F增大逐渐放缓，也就说增大缓存量带来的收益会减小。

设一个投机 token 被 verifier 接受的概率是：

```math
a_p
```

于是：

- 要想第 0 个 lookahead 位置有用，不需要任何前置条件
- 要想第 1 个位置有用，通常前面 1 个位置得先接受
- 要想第 2 个位置有用，前面 2 个位置得先接受
- …
- 所以第 $k$ 个位置被真正"走到"的概率，大致是 $a_p^k$

**也就是说，位置越深，被访问到的机会越小，近似按几何级数下降。**

r是一个常数，对于 $k < K$：

```math
F_k
=
F_0 \cdot a_p^{k/(1+r)}
```

而最末位置 $K$：

```math
F_K
=
F_0 \cdot a_p^{K/(1+r)} \cdot (1-a_p)^{-1/(1+r)}
```

并且 $F_0$ 选到满足预算约束：

```math
\sum_{k=0}^{K} F_k = B
```

满足 capped geometric series（截断几何级数），不做详细推导。

现在可以给出一套 fan-out 策略 $\{F_k^p, F_k^b\}$，算法执行时：

1. 看上一轮 speculation 是 primary 还是 backup 生成的
   - 如果是 primary，用 $\{F_k^p\}$
   - 如果是 backup，用 $\{F_k^b\}$
2. 对每个 lookahead 位置 $k$：
   - 查看 **draft logits**
   - 取 top-$F_k$ 个 token
3. 但要 **排除那个已经被采样并送去验证的 token**
   - 因为那个 token 已经走主验证路径了
   - 缓存里要放的是"备选 outcome"
4. 把这些 top-$F_k$ token 放进 speculation cache
5. 后续如果主验证路径在某处失败，就能直接利用 cache 中预存的备选分支，减少重新计算。

举个例子，假设：

- $K = 4$
- 总预算 $B = 20$
- acceptance rate $a_p = 0.64$
- 幂律指数 $r = 1$

那么：

```math
F_k \propto a_p^{k/(1+r)} = 0.64^{k/2} = 0.8^k
```

于是比例大概是：

```math
F_0 : F_1 : F_2 : F_3 : F_4
\approx
1 : 0.8 : 0.64 : 0.512 : 0.41
```

如果按预算 20 归一化，可能近似得到：

- $F_0 = 6$
- $F_1 = 5$
- $F_2 = 4$
- $F_3 = 3$
- $F_4 = 2$

最后一项再按 theorem 的末项修正，可能会略微调大。

这个算法相比 **每层平均分配** 好在哪里？

如果平均分配：

```math
F_0 = F_1 = \cdots = F_K
```

那会有两个问题：

- 浪费在深层：深层位置很可能根本走不到，放太多候选是浪费 cache
- 前层不够用：前层更容易被访问，却只分到和深层一样的预算，不划算

#### Balancing Cache Hit and Acceptance Rate with Saguaro Sampling

接下来论文发现一个问题：bonus token 很难预测，如何让Target Model给出的 bonus token 更容易落入 Cache 呢？

大多数情况下，投机的 token 都是部分被拒绝又部分被接受，那么在 Not All Accpeted 情况下， bonus token 来自：

```math
r(t) \propto \max(p_{target}(t)-p_{draft}(t),0)
```

target 比 draft 更"想选"的 token，会在 residual 里保留下来，draft 已经过度分配概率质量的 token，不会在 residual 中占优。因此，bonus token 本质上是 "target model 想给、但 draft model 没有给够概率" 的那部分补偿。

所以，bonus token 从 target > draft 的 token 中采样，但 residual distribution 很稀疏，尤其在 **高 temperature **的时候，很难预测这个分布情况。

本文的做法是**主动修改** draft model 给出的 $p{draft} $，从而改变 residual，对于某个 token t，如果把 $p{draft}(t) $ 降低，那么target和draft的差值就会放大，该 token 在 residual distribution 里的权重就更高，从而更容易被选中。

这就是 Saguaro sampling 的核心想法：**故意压低一部分 token 在 draft 中的概率，让它们在 residual 中更容易冒出来。**

不过要兼顾 accpetance rate，因为draft 分布 和 target 分布 越接近，才越容易 Accpet，所以我们只压低 top F 范围内的token，下面看看具体做法。

定义：sampling scheme 是一个把 draft logits 映射到概率分布的函数：

```math
\sigma : \mathbb{R}^V \to \Delta^{V-1}
```

对给定 draft logits $z \in \mathbb{R}^V$，fan-out $F$ ，以及 downweighting 常数 $C \in [0,1]$ ，定义

```math
\sigma_{F,C}(z) \propto
\begin{cases}
C \cdot \exp(z_t), & t \in top_F(z) \\
\exp(z_t), & \text{otherwise}
\end{cases}
```

其中 $top_F(z)$ 是 logits 最大的前 $F$ 个 token，可以看到只对 cache token 降权，但是这样也让被降权的 token 更难被选中了，牺牲了一点"draft 直接采中"的机会，换来"bonus token 更容易被 cache 命中"。

#### Handling Cache Misses with Saguaro Fallback

这一节考虑的是 Cache Miss 的问题，这个 SSD 系统的性能非常依赖 Cache Hit  rate，在一个 batch 中一旦有一个发生 miss，就得重新投机预测，而 batch 中其他 sequence 就会等待。而 batch size 增大时，cache miss 必然发生，

原因很简单，设单个 sequence 的 cache hit rate 为：

```math
p_{hit}
```
batch size = $b$

那么：

整个 batch **全部命中** 的概率：

```math
p_{hit}^b
```

一目了然，batch 调大点，整个 batch 的 命中率 快速逼近 0，论文提出一个简单但非常关键的策略：

- 小 batch 时，用 primary speculator，也就是 慢 但是 准 的 draft model
- 大 batch 时，用 backup speculator，也就是 快 但是 笨 的 draft model

在论文实现里，居然用了 random tokens 做 backup，不明觉厉。

## Discussion

整个 SSD 系统有很拧巴的地方，比如有明显的两点：

- 4.2 节 的 主动修改分布，却声称  SSD 是 lossless 的。
- 4.3 节 的 用 random tokens 做 backup，它的意义不是多生成几个有用 token，而是让系统在 miss 时别阻塞太久。

我们也可以看到 SSD 在高并发场景下 的两个系统瓶颈是：

- speculation cache 的规模随 $B(K+1)F$ 线性增长，带来显存压力
- verification 采用 batch-level 同步，只要任一序列 cache miss，整个 batch 都必须等待 fallback speculation，导致大 batch 时吞吐量急剧下降

还有文章中的很多假设，也未必成立，这部分会在后面继续分析。