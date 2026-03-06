# 论文研读：FlashAttention

论文链接：https://arxiv.org/abs/2205.14135

这篇文章是2022年的了，当时一些问题可能在现在没有那么严重了，但其思想还是很先进的，值得阅读一下。

## Attention

本文只简单地介绍Attention的含义，更聚焦于系统层面的性能加速。

Attention 的作用就是让模型在处理一个位置时，可以动态地查看其他所有位置。

假设一句话：

> The animal didn't cross the street because **it** was too tired.

当模型处理 "it" 时，它需要判断：

- it = animal
- 还是 street？

Attention 的作用就是：让 "it" 自动去关注最相关的词（animal）

核心思想是，对每个 token：

1. 提出一个"问题"（Query）
2. 所有其他 token 提供『信息索引』（Key）
3. 每个 token 携带『信息内容』（Value）
4. 通过相似度决定关注谁
5. 加权汇总信息

数学表达如下：

给定：

$$
Q, K, V
$$

计算：

```math
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```

拆开理解：

第一步：算相似度

```math
QK^T
```

本质是：每个 query 和所有 key 做点积

表示：第 i 个 token 对第 j 个 token 的关注程度

第二步：缩放

```math
\frac{1}{\sqrt{d}}
```

原因：

- 维度越大，点积越大
- softmax 会变得极端
- 缩放可以稳定梯度

第三步：softmax

```math
\text{softmax}
```

把相似度变成概率分布：

- 每一行加起来 = 1
- 表示『关注比例』

第四步：加权求和

```math
\text{weights} \times V
```

意思是：按照关注比例，把所有 value 加权求和得到新的表示。

所谓Self-Attention就是Q/K/V都来自同一个序列，这就叫自注意力。

Multi-Head Attention，即多头注意力，多个head可以：一个head学习语法关系、一个head学长距离依赖...

公式：

```math
\text{MultiHead}(Q,K,V) =
\text{Concat}(head_1,...,head_h)W^O
```

每个 head：

```math
head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

## 性能瓶颈

这篇论文指出了目前Attention计算的一个问题：现代 GPU 的计算速度远超显存带宽，Transformer 大多数操作是 memory-bound。

|     存储     |   带宽    | 容量  |
| :----------: | :-------: | :---: |
| SRAM（片上） |  19 TB/s  | ~20MB |
| HBM（显存）  | 1.5 TB/s  | 40GB  |
|   CPU DRAM   | 12.8 GB/s | >1TB  |

显存访问比片上 SRAM 慢一个数量级。

## 优化方法
TODO