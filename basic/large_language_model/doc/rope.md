# 旋转位置编码

我们知道Transformer的核心就是 **Self-Attention**：

```math
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
```

这个公式有一个重要性质：对 token 的排列是置换不变的，如果把一句话的 token 顺序打乱，只要 embedding 一样，attention 的结果就一样，顺序信息完全丢失。所以必须显式加入位置编码（Position Encoding / Embedding）。

早期的方案是绝对位置编码，对第 $pos$ 个 token，给它一个『位置向量』：

$$
x_{pos} = \text{TokenEmbedding} + \text{PositionEmbedding}_{pos}
$$

这样就附带了顺序信息。

Transformer中使用了Sinusoidal Position Encoding：

对位置 $pos$ 和维度 $i$：

$$
\begin{aligned}
PE(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

为什么要这样设计？这在此不讨论（数学不是本文的主题），我们来看看其中的一个有趣的性质，那就是**『相对位置可线性表示』**。

令：

$$
\mathbf{p}(pos)=
\begin{bmatrix}
\sin(\omega pos)\\
\cos(\omega pos)
\end{bmatrix}
$$

那么位置平移 $k$ 后：

$$
\mathbf{p}(pos+k)=
\begin{bmatrix}
\sin(\omega(pos+k))\\
\cos(\omega(pos+k))
\end{bmatrix}
$$

用三角恒等式展开：

$$
\sin(a+b)=\sin a \cos b+\cos a \sin b
$$

$$
\cos(a+b)=\cos a \cos b-\sin a \sin b
$$

令 $a=\omega pos$, $b=\omega k$，得到：

$$
\mathbf{p}(pos+k)=
\underbrace{
\begin{bmatrix}
\cos(\omega k) & \sin(\omega k)\\
-\sin(\omega k) & \cos(\omega k)
\end{bmatrix}
}_{\mathbf{R}(\omega k)}
\mathbf{p}(pos)
$$

有个有趣的发现：对固定的位移 $k$，矩阵 $\mathbf{R}(\omega k)$ 是一个**固定的旋转矩阵**，所以：

$$
PE(pos+k) = A_k \, PE(pos)
$$

这个性质在每个维度对上都成立，拼起来就是整条位置向量也满足『可由块对角矩阵线性变换得到』。

从"位置 pos 的编码"线性变换一下，就能得到"位置 pos+k 的编码"，因此『相对位移』在表示空间里非常规整（同样的位移 $k$，在所有位置、所有样本中，都以同一种几何变换出现）。

这个性质把『位移』变成了一个模型天然擅长处理的对象——线性变换，从而让 attention 用极低的学习成本就能稳定地感知相对位置，模型只要学会"角度差怎么影响相似度"，就能更容易捕捉相对距离模式。

但是这种编码方式也存在一个显著的问题：位置和语义混在一起。

位置是 **加法** 注入的：

$$
x = token + position
$$

模型要**自己学会**从 embedding 中解析出相对距离，这是困难的，后来就有了相对位置编码。

研究者开始认为，语言建模真正需要的是『相对位置』（前一个、后两个...），而不是"第几个 token"。

定义一个可学习的相对位置偏置矩阵，在 attention score 里直接加相对位置信息：

$$
\text{score}_{ij} = Q_i \cdot K_j + a_{i-j}
$$

其中：

- $a_{i-j}$ 是可学习的相对位置向量

不过这种方式实现相当复杂，这种相对位置编码的典型形式是把相对位置信号直接加进 attention 里，例如：

$$
s_{ij} = Q_i^\top K_j + Q_i^\top r_{(i-j)}
$$

其中 $r_{(i-j)}$ 是相对位移为 $i-j$ 的嵌入。

这意味着模型在计算 **"i 看 j"** 的相似度时，天然就有一个项专门表示 **"它们隔了多少步"**。这就是『直接建模相对位置』的含义，也是优点：**把语言里最重要的结构（相对距离）作为一等公民注入 attention score**，不需要模型绕路自己解码出来。

但要为每一对 (i,j) 准备一个相对位移 embedding，序列长度是 L，那么 pair 有 $L\times L$ 个。

至少要构造一个『相对位置索引矩阵』：

- shape: $[L, L]$
- 每个元素是 $i-j$（或 bucketed 的版本）

然后查表得到相对位置向量：

- 如果相对向量维度是 $d_h$（head dim），那会变成 $[L, L, d_h]$

接着要算：

$$
Q_i^\top r_{(i-j)}
$$

也就是把 $Q$（shape $[L, d_h]$）和 $R$（shape $[L, L, d_h]$）做一次"按 i 对齐"的点积，产出 $[L,L]$。

这会涉及：

- 额外的 embedding lookup
- 额外的张量对齐/扩展（broadcast/reshape/transpose）
- 额外的乘加
- 还要兼容 batch、多头、mask、缓存 KV（推理时更麻烦）

而RoPE的思想是：用一种**旋转**操作，将绝对位置信息以**乘法方式**融入词嵌入，从而在注意力机制中**自然地导出相对位置依赖**。它结合了绝对编码的简洁和相对编码的优越性。 

把向量拆成二维子空间：

对任意向量 $x \in \mathbb{R}^d$，拆成：

$$
(x_0, x_1), (x_2, x_3), \dots
$$

每一对都是一个 **二维平面**，每个二维平面，对应一个频率 $\omega_i$：

$$
\omega_i = 10000^{-2i/d}
$$

在每个二维平面里做旋转，对于位置 $p$，定义旋转角：
$$
\theta_i(p) = \omega_i \cdot p
$$

对任意二维分量 $(x_m,x_n)$，旋转为：

```math
\begin{bmatrix}
x'_m\\
x'_n
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta)\\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
x_m\\
x_n
\end{bmatrix}
```

于是，我们定义位置相关的旋转算子 $R_{\text{pos}}$。
RoPE 做的就是：

$$
Q_i = R_i\,\tilde Q_i,\qquad K_j = R_j\,\tilde K_j
$$

旋转后点积会把 $i,j$ 合并成 $j-i$，这是 RoPE 最核心的一步：

$$
(R_i q)\cdot (R_j k) \;=\; q\cdot (R_{j-i} k)
$$

这里就不详细展开推导这个公式了，可以明显感觉到的地方是：RoPE 不是加，而是『变换』。

RoPE 做的是：

$$
Q_i = R_i \tilde Q_i,\quad K_j = R_j \tilde K_j
$$

然后：

$$
s_{ij} = Q_i^\top K_j
= \tilde Q_i^\top R_{j-i}\tilde K_j
$$

这和相对位置信号的直接相加大不相同，RoPE 把“相对位置”编码成**旋转后的内积**，这是对 attention 最自然、信息损失最小的方式。

