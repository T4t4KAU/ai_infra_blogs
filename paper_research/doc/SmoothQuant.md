# 论文研读：SmoothQuant

论文链接：https://arxiv.org/abs/2211.10438

该文章的标题为《SmoothQuant: accurate and efficient post-training quantization for large language models》

该文章成功发表于ICML 2023（CCF-A），作者分别来自MIT和NVIDIA，代码开源在：https://github.com/mit-han-lab/smoothquant

致以最热烈的赞美和感激！

## 概述

LLM 的激活（activations）一旦变大就很难做 INT8 量化，因为量化（比如 INT8）通常要给一个张量选定范围，再把连续值映射到 256 个离散刻度上。问题在于 **LLM 的 activation 分布在大模型里会出现极少数特别大的值（outliers）**，也就是我们常说的离群值。

现有方法有一些根本缺陷：

|        方法         |        思路        |        问题        |
| :-----------------: | :----------------: | :----------------: |
|     Naive W8A8      |  权重+激活都 INT8  |   大模型精度崩塌   |
|      ZeroQuant      | per-token 激活量化 |     175B 失败      |
|     LLM.int8()      | outlier 保留 FP16  | 速度慢、硬件不友好 |
| Outlier Suppression |   clip / LN 改造   |   只对小模型有效   |

作者的目标是：设计一种 **efficient（高效）+ hardware-friendly（能直接吃到硬件加速）+ training-free（无需再训练）** 的方案，做到 **所有算力重的算子都能用 INT8**，而不是混精/部分算子回退 FP16，一句话概括：**多**（囊括所有compute-intensive算子）、**快**（效率高）、**好**（对硬件友好）、**省**（免训练，低成本）。

## 问题

**为什么activation 中存在严重的 outlier（离群值），直接 INT8 会导致模型崩溃？**

先回顾，激活值是什么？激活值是由前一层的输出，通过线性变换（及可能的非线性变换）所产生的中间张量值。

在神经网络里，以最基本的线性层为例：
```math
Y = XW
```

其中：

- $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$：**权重（weights）**

- $X \in \mathbb{R}^{T \times C_{\text{in}}}$：**输入激活（input activations）**

- $Y \in \mathbb{R}^{T \times C_{\text{out}}}$：**输出激活（output activations）**

上式中 X 和 Y 都属于 activation，即网络在某一层『当前计算状态』的数值表示，

在 Transformer 的一层中，activation 包括但不限于：

- Token embedding
- Q / K / V 的投影结果
- Attention score（QKᵀ）
- Attention 输出
- FFN 中间层输出

例如在自注意力中：

```math
Q=XW_Q,\space K=XW_K,\space V=XW_V
```

这里：

- $X$：上一层的 activation
- $Q, K, V$：**新的 activation**

这些 activation，在推理阶段大量生成，数量远多于权重，必须被量化或缓存（如 KV cache）。

再看，INT8 量化是怎么工作的？

量化的本质是将连续的高精度数值（FP32 / FP16）映射到有限个离散值（INT8 / INT4 等），并在可接受误差内近似原始计算。



最常见的是均匀对称量化，是 **INT8 GEMM 硬件最友好** 的形式，即假设数值在0处对称，对于给定浮点张量有 $X$：

```math
\bar{\mathbf{X}}^{\mathrm{INT}8}=\lceil\frac{\mathbf{X}^{\mathrm{FP}16}}{\Delta}\rfloor,\quad\Delta=\frac{\max(|\mathbf{X}|)}{2^{N-1}-1}
```
存在两个严重的问题：

- 截断误差
- **尺度失配**（由极值决定，LLM activation 的关键问题）

我们还要额外考虑量化粒度。量化粒度指的是，一个量化 scale（Δ）作用在张量的多大子集上，有三种量化粒度，下面逐一讨论。

我们假设 activation 张量是：
```math
X \in \mathbb{R}^{T \times C}
```
其中：

- $T$：token 数（行）
- $C$：channel 数（列）

**Per-tensor quantization **就是按『整个张量』量，整个张量只用一个 scale：

```math
\Delta = \frac{\max_{t,c}|X_{t,c}|}{127} 
```

这个方式是最容易实现的，这种情况下，一个scale作用的范围是最大的， 任何一个 outlier 都是灾难性的。

**Per-token quantization** 每个 token（一行）有自己的 scale：
```math
\Delta_t = \frac{\max_c|X_{t,c}|}{127}
```
解决了不同 token 整体幅值不同的问题，但同一 token 内 channel 差异巨大，一行里如果有 1 个 channel 是 outlier，那么这一行的scale都会被拉到很大。

**Per-channel quantization** 每个 channel（一列）有自己的 scale：
```math
\Delta_c = \frac{\max_t|X_{t,c}|}{127}
```
这种情况下，outlier 只影响它自己的 channel，不会影响其他 channel，但没法和 GPU 上的高性能矩阵乘（GEMM）很好地配合。

因为在 GPU 上，真正快的矩阵乘是靠：

- **Tensor Core MMA 指令**
- 一条指令做一小块矩阵乘加
- 流水线极深、吞吐极高

这样就要求一整段计算全是『同一类型的高吞吐指令』，不能被慢指令打断。如果 activation 是 **per-channel 量化**，就意味着activation 的每一列（channel）都有 **不同的 scale**，但Tensor Core 无法在内积过程中，对每一项乘法插入不同的 scale，否则就无法享有理想的硬件加速，但是量化可以被用在外维乘积上。

如下式所示，缩放被放到外维上：

```math
\mathbf{Y}=\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{X}}^{\mathrm{FP}16})\cdot(\mathbf{\bar{X}}^{\mathrm{INT}8}\cdot\mathbf{\bar{W}}^{\mathrm{INT}8})\cdot\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{W}}^{\mathrm{FP}16})
```

中间这一项：

```math
\mathbf{\bar{X}}^{\mathrm{INT}8}\cdot\mathbf{\bar{W}}^{\mathrm{INT}8}
```

是高吞吐的矩阵运算，并行地放在硬件上执行，应享受最大程度地硬件加速，不应被打断。

左边的是对角矩阵activation scaling，对应per-token activation scale：

```math
\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{X}}^{\mathrm{FP}16})
```

这一项是在完成中间的 GEMM 之后再乘，这是外维（token 维）缩放。

右边的对角矩阵（weight / output scaling），对应per-output-channel scale：

```math
\mathrm{diag}(\Delta_W^{\mathrm{FP16}})
```

每一列一个 scale，同样在 GEMM 之后做， 是外维（输出通道维）缩放。可知，所有 scaling 都发生在矩阵乘法的『外维』，而不是『内积维』，外维是可以缩放的。

如果做per-channel 量化，实际上会变成：

```math
Y=(\bar{X}\cdot\mathrm{diag}(\Delta_{X,\mathrm{channel}}))\cdot W
```

这意味着scaling **插在 GEMM 内部**，必须要沿着**内积维**在每次乘加时应用不同 scale，这对硬件来说是极不友好的。

如下图所示，清晰地展现了量化粒度各自的特性：
![SmoothQuant_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/SmoothQuant_1.png)

量化粒度决定了谁来『决定最大值』：

- per-tensor：**全局最大值**
- per-token：**每行最大值**
- per-channel：**每列最大值**

最大值决定了 scale，scale 决定了误差，同一 token 内，少数 channel 极大。

接下来，分析同一token中为什么会出现outlier？outlier不是偶然出现的，而是一个结构性问题，因为：

- 多层线性变换不断叠加

- 注意力机制会**放大特定方向的响应**

- FFN 中的大矩阵进一步放大某些特征方向

所以**极少数特征方向（channel）”会被反复强化**，论文指出outlier **集中在固定的一小部分 channel**，在不同 token、不同样本中反复出现，幅值比其他 channel **大 10～100 倍**，这不是偶然发生的，而是模型学到的、**有功能意义的激活模式**。

来到我们最终的问题：outlier 是如何毁灭INT8量化的？

首先，outlier 拉大 了scale，假设某一行（一个 token）的 activation：

- 大多数 channel： $\approx 0.1$
- 一个 outlier channel： $\approx 100$

那么：
```math
\Delta = \frac{100}{127} \approx 0.79
```

而对普通 channel：
```math
0.1 / 0.79 \approx 0.13 \;\Rightarrow\; \mathrm{round}(0.13) = 0
```
结果绝大多数 channel 变成0了，只剩下 outlier channel 有非零值，也就是**low effective bits**，并且误差会在层与层之间**级联放大**，接着发生注意力权重会被错误计算和FFN 输出方向发生系统性偏移，最终导致模型不能再用。

**为什么 weight 没有这个问题？**

weight 是训练后固定的，分布更平滑，更接近零均值，**几乎没有极端 outlier**。

具体而言，weight 是在强正则、梯度约束和参数共享下，通过长期优化学到的『稳定统计量』，而 activation 是输入驱动、逐层放大、无显式约束的中间响应，因此前者分布平滑、后者容易出现极端值。

在训练中：

- **weight $W$** 是优化目标
- **activation $X$** 只是中间计算结果

优化问题本质是：

$$
\min_W \;\mathbb{E}_{(x,y)}[\mathcal{L}(f_W(x), y)]
$$

可见只有 **weight 会被梯度直接约束**，梯度下降天然抑制极端 weight，设某个 weight $w_i$ 变得非常大，会放大输出放大 loss，放大反向传播梯度，也就是极端大的 weight 会产生不成比例的梯度惩罚，所以极大的outlier会被SGD拉回。

并且在实际训练中，几乎所有大模型都使用：

- **weight decay（L2 正则）**
- Adam / AdamW 等自适应优化器

以 L2 正则为例，目标函数变为：

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \|W\|_2^2
$$

对每个 weight：

$$
\nabla w_i \;\supset\; 2\lambda w_i
$$

weight 越大，惩罚越大，极端值被持续压回。

参数共享会进一步『均化』weight 分布，Transformer 中，同一套weight会被所有token反复使用，所以任何对少数样本有利、但对整体不稳定的极端 weight，都会在总体 loss 中被淘汰。

**过去的量化方法有什么问题？**

Naive W8A8的问题是：它假设激活分布是『均匀』的，不过这一点被证实是谬误的，模型越大则 outlier channel 出现概率越高。

ZeroQuant使用的是per-token quant，具体问题已经讲过，没有解决『channel 间失衡』。

LLM.int8()选择不量化outlier，超过阈值的部分依然使用FP16，其余正常值用INT8，但会对硬件极不友好，一个 GEMM 被拆成：INT8 GEMM和FP16 GEMM，导致Tensor Core 无法高效融合，需要频繁在 INT8 ↔ FP16 间切换，吞吐大幅下降。

Outlier Suppression 选择直接将outlier压下来（截断、平滑、归一化改造），但实际上LLM 的 outlier 不是噪声，而是『有功能的信号』，通道级 outlier 往往是模型表示中的**关键方向**，某些 channel 被训练成『强特征探测器』，在很多 token 上反复出现大幅值（systematic），如果直接抑制，相当于把模型的某些关键特征强行削弱，小模型或许尚可，但大模型的表示更精细，某些通道承担的语义更明确，你一刀切地压制，会系统性伤害这些语义方向。

## 方法

本文有一些关键发现：

Activation 的 outlier数量少且**集中在固定 channel**，如果一个channel出现了一个outlier，则所有的token都会在这个channel出现outlier；并且，一个token内的不同channel之间的方差很大，但不同 token 在同一 channel 上幅值相近，故而量化难度来自 channel 之间的不平衡。

从这一点思考，做per-channel量化会更加适合，所以作者试图寻求一个手段，使得channel之间更加"平滑"，从而更加容易量化。

于是，文章提出，对线性层（或 matmul）：
```math
Y = XW
```
SmoothQuant 做一个**数学等价变换**：对输入通道维度（ $C_i$ ）引入一个对角缩放 `diag(s)`：
```math
Y = (X\mathrm{diag}(s)^{-1})\cdot (\mathrm{diag}(s)W)=\hat X\hat W
```

其中：

- $\hat X = X\mathrm{diag}(s)^{-1}$（激活按通道除以 $s$ ，变平滑）
- $\hat W = \mathrm{diag}(s)W$（权重按同样通道乘以 $s$ ，保持等价）

这是**把激活通道间的尺度差异（outlier）挪到权重上**，因为权重量化相对更稳，因为这个变换不需要训练参数； $s$ 用少量校准样本统计就行，而且可以把 $s$ **融合到前一层（如 LayerNorm/Linear）的参数里**，让运行时不额外加 kernel，核心思想就是把激活的量化难度转移到权重中（*Migrate the quantization difficulty from activations to weights*）。

如下图所示：

![SmoothQuant_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/SmoothQuant_2.png)

现在如何选择 $s$ 是一个问题， $s$ 如果太大虽然能拉平激活，但是权重却被放大了，这也导致量化难以进行。

因此引入超参 **迁移强度 α**（0~1）在两者之间折中：

```math
s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
```

其中 $j$ 是输入通道的索引，可见：

- α 越大：更多"难度"从激活迁到权重（更强 smoothing 激活）
- α 越小：保守，更多"难度"留在激活

这个α应该如何设置，论文给出经验：

- 对 OPT/BLOOM，α=0.5 往往是均衡点
- 对激活 outlier 更严重的 GLM-130B，用更大 α（比如 0.75）更好
- 也做了消融：α 太小激活难量化；α 太大权重难量化；sweet spot 大概在 0.4~0.6（OPT-175B 实验）

如下图展示了α=0.5的情况：

![SmoothQuant_3](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/SmoothQuant_3.png)

在α=0.5的情况下，刚好将激活和权重都调整到相同的范围，因此它们在量化时承受**相近的难度/误差压力**，具体可作如下推导:

变换后 activation 的最大值：

```math
s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
```

整理后得到：

```math
\max|\tilde X_j|
= \max(|X_j|)^{1-\alpha} \cdot \max(|W_j|)^{1-\alpha}
```

变换后 weight 的最大值：
```math
\max|\tilde W_j|
= s_j \cdot \max|W_j|
= \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}} \cdot \max(|W_j|)
```
整理后得到：
```math
\max|\tilde W_j|
= \max(|X_j|)^{\alpha} \cdot \max(|W_j|)^{\alpha}
```
故当 α 取中间值（如 0.5）时：
```math
\max|\tilde X_j|
\;\approx\;
\max|\tilde W_j|
\;\approx\;
\sqrt{\max(|X_j|)\cdot\max(|W_j|)}
```
也就是说：**在同一个 channel 上，变换后的 activation 和 weight 的最大幅值被"拉到同一个量级"**，这样做的好处就是避免任何一方在矩阵乘中独占误差来源，使 INT8 的有限刻度在两侧都被充分利用，从而将原本由 activation outlier 主导的灾难性误差，转化为可控、可叠加的量化噪声，模型因此不再崩溃。

下面举个例子看看。

我们先假设激活值如下（第8个是outlier）：

```math
X = [ 0.08, 0.12, 0.05, 0.09, 0.11, 0.07, 0.10, 100.0 ]
```

那么activation 的scale是：
```math
\Delta_X = \frac{\max |X|}{127}
= \frac{100}{127}
\approx 0.787
```
量化结果：

| channel | 原值 | 原值 / Δ | INT8 |
| :-----: | :--: | :------: | :--: |
|    1    | 0.08 |   0.10   |  0   |
|    2    | 0.12 |   0.15   |  0   |
|    3    | 0.05 |   0.06   |  0   |
|    4    | 0.09 |   0.11   |  0   |
|    5    | 0.11 |   0.14   |  0   |
|    6    | 0.07 |   0.09   |  0   |
|    7    | 0.10 |   0.13   |  0   |
|    8    | 100  |   127    | 127  |

结果激活被量化成：

```math
X_{int} = [0, 0, 0, 0, 0, 0, 0, 127]
```

如上文所述，量化后 **只剩 1 个有效维度**，其余 7 个 channel 的信息 **完全丢失**，activation 被 outlier 独占量程，只用了127这一个有效刻度。

假设权重如下：

```math
W=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09]
```

代入 $\alpha=0.5$：
```math
s_j=\frac{|X_j|^{0.5}}{|W_j|^{0.5}}=\sqrt{\frac{|X_j|}{|W_j|}}
```
代入上面 $s_j$ 的形式，我们会发现一个很漂亮的性质（α=0.5）：
```math
\tilde X_j = \sqrt{|X_j||W_j|}\cdot \mathrm{sign}(X_j)
,\qquad
\tilde W_j = \sqrt{|X_j||W_j|}\cdot \mathrm{sign}(W_j)
```
也证实了：同一通道上，变换后的激活和权重的最大值会被拉到同一个量级（几何平均）。

我们可以计算出如下结果：

| $j$  | $X_j$ | $W_j$ | $s_j=\sqrt{X_j/W_j}$ | $\tilde X_j=X_j/s_j$ | $\tilde W_j=s_jW_j$ |
| :--: | :---: | :---: | :------------------: | :------------------: | :-----------------: |
|  1   | 0.08  | 0.10  |        0.894         |        0.089         |        0.089        |
|  2   | 0.12  | 0.10  |        1.095         |        0.110         |        0.110        |
|  3   | 0.05  | 0.10  |        0.707         |        0.071         |        0.071        |
|  4   | 0.09  | 0.10  |        0.949         |        0.095         |        0.095        |
|  5   | 0.11  | 0.10  |        1.049         |        0.105         |        0.105        |
|  6   | 0.07  | 0.10  |        0.837         |        0.084         |        0.084        |
|  7   | 0.10  | 0.10  |        1.000         |        0.100         |        0.100        |
|  8   | 100.0 | 0.09  |        33.333        |        3.000         |        3.000        |

接着计算出scale：
```math
\Delta_{\tilde X}=\frac{\max|\tilde X|}{127}=\frac{3}{127}=0.0236
```
量化结果是：
```math
\tilde X_{\text{int}}\approx[4,5,3,4,4,4,4,127]
```
很明显，这次用到了更多的刻度，权重也几乎没影响，这只是8 channel，如果是4096个channel会铺的更开。

要注意的是，激活值是受到输入样本影响的，不是固定的，作者认为并不用知道每次推理时的『真实激活最大值』，只需要用一小批代表性数据，离线估计『激活在统计意义上的尺度』，然后把缩放永久吸收到权重里，在线推理时完全不依赖激活的真实最大值。

另外要补充的是，SmoothQuant **只对 Linear 层的输入 activation 做 smoothing**

也就是只处理如下情况：

- Attention 里的 Q/K/V 输入
- FFN 里的 FC1 输入

因为Linear 是量化瓶颈，而非 Linear 算子（Softmax/LayerNorm）本身不适合 INT8

如下图所示：

![SmoothQuant_4](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/paper_research/img/SmoothQuant_4.png)

标绿的算子被看成Transformer Block中属于compute-intensive算子，是应该被量化的部分。

SmoothQuant 的方法在形式上很简单，但在问题建模、切入点选择和工程可落地性上并不简单，SmoothQuant 做的事只有：
```math
XW = (X D^{-1})(D W)
```
再搭配一个经验参数 $\alpha$，看似简单，不过也依赖几个重要前提：

- activation outlier 通道是稳定的，outliers集中出现在固定通道

- weight 分布相对平滑

- 线性算子主导计算

在 **Transformer-based LLM** 这个主流场景下，这些前提**恰好长期成立**。

过去的方法有的复杂，有的粗暴，基本围绕：

- scale 粒度（per-token / per-group）

- outlier 检测（threshold）

- 混精绕行（LLM.int8）

- 分布压缩（clip）

SmoothQuant不强行压制问题，而是通过等价重构顺应系统本性，让原本冲突的约束自然达成平衡。

这似乎很符合道家的思想：*天之道，损有余而补不足*

## 实验复现

TODO
