# 论文研读：FlexGen

论文链接：https://arxiv.org/abs/2303.06865

## 概述

大模型在推理时显存需求非常大，OPT-175B 的 FP16 权重 alone 就要 325GB 显存。传统做法是使用多张高端GPU，或者依赖 DeepSpeed / Accelerate 的 offloading技术。

但这些offloading推理系统存在：

- batch size 极小
- 吞吐量低
- 在单张消费级 GPU（如 T4 16GB）上几乎不可用

该论文关注的场景是吞吐为导向的生成式推理，不针对聊天机器人这种实时场景。该方法试图解决：如何在『只有一张 16GB GPU』的条件下，高吞吐地运行 175B 级 LLM 推理？

难点有三点：

- 显存远远不够
- KV Cache 比模型权重还大（大 batch 时尤其严重）
- 传统 offloading 方法存在弊端：
  - 调度方式来自「训练」，照搬了训练时期的offloading调度
  - 完全不适合「生成式推理」

本文的方法：**把 GPU / CPU / 硬盘 当成一个统一的"多级内存系统"，为『生成式推理』重新设计调度、存储和压缩策略。**

## 问题

**为什么说训练使用的offloading调度方式不适合生成式推理？**

训练关心的是「一次前向 + 反向能不能尽快结束」，生成式推理关心的是「权重能不能少搬、KV Cache 能不能撑大 batch」

两者的目标就不一样，所以照搬训练用的调度方式是不合理的。

Transformer训练中的典型调度方式(row-by-row)：

```
样本 1：layer1 → layer2 → ... → layerL
样本 2：layer1 → layer2 → ... → layerL
```

训练中的一个 iteration 通常包含：

- **Forward**：输入序列（长度 s）一次性喂给模型，层层计算得到 logits

- **Loss**：与 label 计算损失（交叉熵）

- **Backward**：反向传播，逐层计算梯度

- **Optimizer step**：更新参数（Adam/SGD 等）

特点：

- forward 只做一次（对这批数据而言）
- backward 必须访问 forward 期间产生的中间量（激活）
- 训练的核心是『梯度计算 + 参数更新』

在训练中，激活值占用空间非常大，权重尽量常驻GPU，如果空间不够可以offload一部分激活值或者优化器状态，每一层权重只用一次就不用了。

但是生成推理就并非如此了，生成式推理分为两段：

- **Prefill**：把 prompt（长度 s）一次性跑完，建立每层的 KV Cache
- **Decode**：每生成一个 token，都要再跑一遍所有层，但每层注意力会复用 KV Cache，并在末尾追加新 token 的 KV

如下图所示，横轴是 token 时间，纵轴是 batch 中的样本，每个小方块是『某一层的一次计算』：

![flexgen_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/flexgen_1.png)

图中颜色不同的小方块，表示不同的 transformer layer，同一颜色表示同一层，即共享同一组权重。

同一列（同一个 token）里：

- 不同 batch 样本彼此独立
- 可以并行算

水平方向箭头：

- 同一个 token，在网络中逐层前向传播

- 表现了标准 transformer 的 layer-by-layer 依赖

如果还使用row-by-row方式：

```
token 1:
  layer1（load 权重）
  layer2（load 权重）
  ...
token 2:
  layer1（再 load 一次）
  layer2（再 load 一次）
```

同一层权重，被反复从 CPU / Disk 拉到 GPU，导致GPU 大部分时间在等 I/O，论文明确指出这一点是现有系统（DeepSpeed / Accelerate）吞吐极低的根本原因。

在训练中，KV Cache不会被考虑到，但在推理中，大batch下的KV Cache比模型权重还大，训练调度完全没考虑，KV Cache 生命周期、KV Cache 存放位置，KV Cache 的 I/O 成本。 于是，这直接限制了 batch size的大小。

训练和推理的差异，可作如下总结：

|          |  训练  |  生成推理  |
| :------: | :----: | :--------: |
| 优化目标 | 单步快 | 总 token/s |
|  batch   |  中等  |  越大越好  |
|   权重   | 用一次 |  用 n 次   |
| KV Cache |   无   |    极大    |
|   I/O    |  次要  |  核心瓶颈  |

所以说，训练用的调度方式不能直接用于推理。本文的目标是找到一条有效路径，以最小化总执行时间，这包括在设备间移动张量时的计算成本和I/O成本。

所以本文采用了"之字型"调度，如下图所示：

![flexgen_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/flexgen_2.png)

图中的block是多个batch的集合，上面已经讲了row-by-row的劣势，而zig-zag的调度是：

```
固定 layer 权重
  ↓
在 block 内，对多个 batch 连续计算
  ↓
再换下一层权重
```

这样实现了最大程度的**权重复用**，加载一层权重后就算完block内所有的batch，可作如下总结：

|    维度    |   Row-by-row    | Zig-zag block |
| :--------: | :-------------: | :-----------: |
|  遍历主序  |  样本 / token   | layer / block |
|  权重使用  |   用一次就扔    | 用一次狠狠干  |
|  权重 I/O  | ∝ token × batch | 被 block 摊薄 |
|  KV cache  |    被动管理     |   显式建模    |
| GPU 利用率 |      极低       |   显著提高    |
|  适合场景  |  训练 / 小模型  |  大模型推理   |

**为什么在某些情况下，把注意力计算放到 CPU 上反而更快？**

这有点反直觉，当 KV cache 在 CPU 上、且序列很长时，把 attention 计算交给 CPU，居然比把 KV cache 搬到 GPU 再算，更快、更省带宽。

FlexGen 讨论的典型场景是：GPU 显存不够 ，KV Cache 存到 CPU 内存。

所以，如果在GPU 上算 attention，真实系统会：

1. KV cache 在 **CPU 内存**
2. GPU 要算 attention，必须**把 KV cache 从 CPU 拷到 GPU**
3. GPU 计算 attention
4. 继续后续层计算

这样的话，KV Cache 每个 decode step 都在增长，但每步都要读完整历史，会产生巨大的I/O量。所以GPU算再快也没用，因为瓶颈在 **PCIe / NVLink**：

```
等 KV cache 传完 → 0.1ms
算 attention → 0.01ms
```

搬运的数据量足有：

```math
\underbrace{b}_{batch}\times\underbrace{s}_{sequence}\times\underbrace{h_1}_{hidden}\times\underbrace{4}_{FP16的K+V}\quad\mathrm{bytes}
```

改成 CPU 计算 attention 后：

1. KV cache **本来就在 CPU**
2. GPU 只把 **Q（activation）** 传给 CPU
3. CPU 用本地内存算 attention
4. 把结果返回 GPU（或继续在 CPU）

在这个方案下，只要搬运一个Q即可，和序列长度无关。

只用搬运：

```math
b\times h_1\times4\quad\mathrm{bytes}
```

## 方法

### 成本模型

本文提出了一个搜索空间并开发了一个成本分析模型，以根据这些算法参数和硬件规格估算执行时间。这个 Cost Model 要做的是：给定硬件带宽/算力 + 给定策略参数，快速估计『每个 block 的执行时间 T』，并加入内存约束，方便搜索最优策略。

本文构建了一个 **分层（layer-level）的延迟模型**，把一次推理拆成：

- **Prefill（处理 prompt）**
- **Decoding（逐 token 生成）**

并且把 **计算 + CPU/GPU/磁盘 I/O** 全部纳入，最终用来：

- 预测总延迟
- 检查 GPU / CPU / 磁盘内存是否超限
- 在不同调度 / 放置策略中做搜索和选择

计算单个 block 的总延迟：

```math
T=T_{pre}\cdot l+T_{gen}\cdot(n-1)\cdot l
```

其中：

|   符号    |            含义             |
| :-------: | :-------------------------: |
|    $l$    |      Transformer 层数       |
|    $n$    |     需要生成的 token 数     |
| $T_{pre}$ | prefill 阶段：单层平均延迟  |
| $T_{gen}$ | decoding 阶段：单层平均延迟 |


可见：

- **Prefill**: prompt 只跑 **一次**，但要过**所有层**

- **Decoding**: 每生成一个 token，都要跑**所有层**，第一个 token 已经在 prefill 中算过了，所以是 n - 1

接着，本文提出了一个关键假设：**CPU↔GPU↔Disk 的数据传输可以和计算并行**

于是，**总延迟 = 最慢的那一项**，那么Prefill 阶段的单层延迟是:

```math
T_{pre}=\max(ctog^p,\mathrm{~}gtoc^p,\mathrm{~}dtoc^p,\mathrm{~}ctod^p,\mathrm{~}comp^p)
```

其中：
|    项    |         含义          |
| :------: | :-------------------: |
| $ctog^p$ |    CPU → GPU 拷贝     |
| $gtoc^p$ |    GPU → CPU 拷贝     |
| $dtoc^p$ |    Disk → CPU 读取    |
| $ctod^p$ |    CPU → Disk 写入    |
| $comp^p$ |       计算时间        |
| 上标 $p$ | 表示 **prefill 阶段** |

Decoding 阶段的单层延迟 $T_{gen}$:

```math
T_{gen}=\max(ctog^g,\mathrm{~}gtoc^g,\mathrm{~}dtoc^g,\mathrm{~}ctod^g,\mathrm{~}comp^g)
```

和 prefill 完全一样，只是：

- 上标 $g$ = generation（decoding）
- KV cache 行为、激活大小、访问模式不同

论文指出可以汇总所有的I/O事件估算出延迟，下面单独推导一下 $dtoc^p$ ，即Disk到CPU的传输延迟。

权重在FP16 情况下：

```math
weights \space size=8h_1^2+4h_1h_2\mathrm{~bytes}
```

其中：

- $h_1$：hidden size

- $h_2$：MLP 第二层 hidden size

激活大小：

```math
activations \space size=2\cdot bls\cdot h_1
```

其中：

- $bls$：block size

- 与 batch / token 数有关

KV Cache（平均）大小：
$$KV size = 4 \cdot bls \cdot (s + \frac{n}{2}) \cdot h_1$$
其中：
- $s$：prompt length
- $\frac{n}{2}$：平均生成长度（经验估计）

本文假设不一定所有数据都在磁盘，只需加载 **一定比例**，可设：

| 参数 |             含义             |
| :--: | :--------------------------: |
| $wd$ |   weights 从 disk 加载比例   |
| $hd$ | activations 从 disk 加载比例 |
| $cd$ |  KV cache 从 disk 加载比例   |

 最终推算出Disk 到 CPU的I/O 延迟公式：
```math
dtoc^g =
\frac{
(8h_1^2 + 4h_1h_2)\cdot wd
+ 4\cdot bls\cdot(s+\frac{n}{2})\cdot h_1\cdot cd
+ 2\cdot bls\cdot h_1\cdot hd
}{
disk\_to\_cpu\_bandwidth
}
```
这个公式是在定量估计 decoding 阶段「磁盘 → CPU」的I/O 延迟，本质是把三类数据的大小加起来，再除以磁盘带宽：
```math
time \approx \frac{bytes \space to \space read/write}{disk \space bandwidth}
```
以上讨论了怎么估计dtoc，估算ctog / gtoc 也是同理，主要基于：
```math
ctog \approx \frac{bytes \space Host→Device}{H2D \space bandwidth}
```
```math
gtoc \approx \frac{bytes \space Device→Host}{D2H \space bandwidth}
```

两种情况会发生GPU和CPU之间的传输：

- CPU 上暂存的 weights / KV / activations 被搬到 GPU
- GPU 产出的中间结果被搬回 CPU（一般推理里尽量避免）

而估算comp，主要基于最常用的分析估算：
```math
comp \approx \sum_i \frac{FLOPs_i}{effective \space {throughput_i}}
```
另外，该模型还加入了峰值内存约束，防止搜索出来的策略导致系统发生OOM，即使性能最优。

在论文的附录A.3中给出了具体的计算公式

### 策略搜索

现在要将策略抽象成策略参数，便于做策略搜索。策略搜索要做的是：在不超过 GPU / CPU / 磁盘内存约束的前提下，最大化生成式推理的吞吐量（token/s）。

FlexGen 最终把策略写成一个 **policy**，包含 11 个变量：

1. block size：`bls`
2. GPU batch size：`gbs`（一个 block 内拆成多少个 GPU micro-batches）
3. 权重放置比例：`wg, wc, wd`（GPU/CPU/disk）
4. activation 放置比例：`hg, hc, hd`
5. KV cache 放置比例：`cg, cc, cd`

其中前两项是离散变量，后三项是连续变量。虽然实际系统中的张量是不可无限切分的，所以不应表示为实数，但本文放宽了条件。

策略搜索基于两层：

- 外层是枚举 `(bls, gbs)`（小规模离散搜索），列举(bls, gbs)的组合。文章指出，gbs通常是是4的倍数，bls小于20，所以选择不多
- 对每一个 `(bls, gbs)`，在 placement 变量空间$p = (wg, wc, wd, cg, cc, cd, hg, hc, hd)$中，找『最快、且不 OOM』 的解。

接下来，使用线性规划方法找到符合条件的最优解，其约束如下：

-  placement 归一化约束：

  ```math
  wg + wc + wd = 1
  ```
  ```math
  hg + hc + hd = 1
  ```
  ```math
  cg + cc + cd = 1
  ```

- 峰值内存约束：

  ```math
  gpu \space peak \space memory < gpu \space mem \space capacity
  ```
  ```math
  cpu \space peak \space memory < cpu \space mem \space capacity
  ```
  ```math
  disk \space peak \space memory < disk \space mem \space capacity
  ```

- 非负性约束：

  ```math
  wg, wc, wd ≥ 0
  ```
  ```math
  hg, hc, hd ≥ 0
  ```
  ```math
  cg, cc, cd ≥ 0
  ```

优化目标：

```math
\min\frac{T(bls,gbs,w,h,c)}{bls}
```

其中，T由cost model给出，即处理一个block的总时间。

这个LP问题是可解的，在此不作详细证明。

整个流程可以归纳为：

```
best_policy = None
best_throughput = 0

for bls in candidate_block_sizes:
  for gbs in candidate_gpu_batch_sizes:

    构建 LP:
      变量：wg,wc,wd,hg,hc,hd,cg,cc,cd
      目标：min T(bls,gbs,...) / bls
      约束：
        - placement 归一化
        - GPU/CPU/disk 峰值内存
        - 非负

    求解 LP

    if LP 可行:
        throughput = bls / T
        if throughput > best:
            更新 best_policy
```

应当能搜索这样的解：

```
bls = 128
gbs = 4

wg = 0.0   wc = 0.5   wd = 0.5
hg = 0.2   hc = 0.8   hd = 0.0
cg = 0.0   cc = 1.0   cd = 0.0
```

### 多卡并行

本文指出，多 GPU 不只是线性加速，甚至可能出现『超线性加速』。

本文首先指出，即使在单 GPU 情况下：

- 已经用 zig-zag block
- 已经把权重 / KV cache / activation 合理 offload
- 已经 overlap I/O 和 compute

但瓶颈仍然存在：

- CPU ↔ GPU / Disk ↔ CPU 的 I/O 有很高的延迟
- GPU 不能一直满载计算

但是多 GPU + 多 CPU 会改变局面，因为：

- 模型被切分到 m 张 GPU

- 每张 GPU 只负责 1/m 的模型

结果是：

- 每张 GPU 的 权重更少
- 每张 GPU 的 KV cache 更少
- 更多数据可以留在 GPU 上

所以，offloading 的压力下降了， I/O 时间下降得比 GPU 数量更快，GPU 利用率反而上升，于是带来了超线性的加速。

值得注意的是，FlexGen只选择PP这一种并行策略，不用TP的原因是：通信频繁，在 offloading + I/O-bound 场景下，通信会进一步恶化瓶颈。所以文章强调，该系统不适用于实时、强调交互的场景。

在多卡场景下，可以直接复用单 GPU 的策略搜索，如果模型有 $l$ 层，pipeline 有 $m$ 段，那么每段只包含 $l/m$ 层。每个 GPU 上实际运行的子模型，就是一个 **层数更少** 的 Transformer。除了『层数变少』，其余结构（每层的线性层、attention以及KV cache 形状）是一样的。

在单 GPU 模型里：

- 权重总量：
  $$W_{total} \approx l \cdot W_{layer}$$
- KV cache 总量：
  $$C_{KV} \approx b \cdot l \cdot (s+n) \cdot h_1$$
- 激活在某阶段的量：也是"每层/每步近似可加"的

当把模型切成 $m$ 段时，每段只负责 $l/m$ 层：

- stage 权重：
  $$W^{(stage)} \approx (l/m) \cdot W_{layer}$$
- stage KV：
  $$C_{KV}^{(stage)} \approx b \cdot (l/m) \cdot (s+n) \cdot h_1$$

可见， 所有『内存压力项』几乎都是按层数线性缩放，因此把 $l$ 换成 $l/m$ 就能复用单卡的内存约束结构。模型拆分后，内存约束形式不变，只是系数变小，所以可以直接复用单卡策略。

## 实验复现
TODO
