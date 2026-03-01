# LLAISYS ASSIGNMENT 2

提交链接：https://github.com/T4t4KAU/llaisys/commit/b32982da143982febd433dbad9c83e54a04ecad4

这里要求实现一些基本的算子。

## TASK 2.1 Argmax

功能是获取张量`vals`的最大值及其索引，并分别存储在`max_val`和`max_idx`中。

这里给出核心代码：

```c++
void argmax_(int64_t &max_idx, T &max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        max_idx = -1;
        return;
    }

    max_idx = 0;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float best_val = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; i++) {
            float current = llaisys::utils::cast<float>(vals[i]);
            if (current > best_val) {
                best_val = current;
                max_idx = static_cast<int>(i);
            }
        }
        max_val = llaisys::utils::cast<T>(best_val);
    } else {
        T best_val = vals[0];
        for (size_t i = 1; i < numel; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                max_idx = static_cast<int>(i);
            }
        }
        max_val = best_val;
    }
}
```

找到最大值就是一个简单的比大小，要注意的是如果类型是BF16或者FP16就转化成Float进行计算。

对应的PyTorch示例：

```python
x = torch.tensor([[1, 5, 7],
                  [3, 2, 6]])

# 沿着行方向（dim=0）找最大值
values, indices = torch.max(x, dim=0)
print(values)    # tensor([3, 5, 7])
print(indices)     # tensor([1, 0, 0])  # 每列最大值所在的行

# 沿着列方向（dim=1）找最大值
values, indices = torch.max(x, dim=1)
print(values)    # tensor([7, 6])
print(indices)     # tensor([2, 2])  # 每行最大值所在的列

# 使用dim=-1（最后一个维度）
values, indices = torch.max(x, dim=-1)  # 等同于dim=1
print(values)    # tensor([7, 6])
print(indices)     # tensor([2, 2])
```

## TASK 2.2 Embedding

功能是从`weight`（2-D）中复制`index`（1-D）中的行到`output`（2-D）。

`index`必须是Int64类型（PyTorch中int的默认数据类型）。

核心代码：

```c++
void embedding_(std::byte *out, const int64_t *index, const std::byte *weight, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        std::memcpy(out + i * d, weight + index[i] * d, d);
    }
}
```

遍历每个index，将weight中对应的行复制追加到out即可，用一个简单的内存拷贝。

对应PyTorch示例：

```python
# 创建一个嵌入矩阵 (词汇表大小=10, 嵌入维度=5)
embedding_matrix = torch.randn(10, 5)

# 要查找的索引
indices = torch.tensor([1, 3, 5, 2])

# 查找嵌入
embedded = F.embedding(indices, embedding_matrix)

print(embedding_matrix)
print(f"输入索引: {indices}")
print(f"嵌入矩阵形状: {embedding_matrix.shape}")
print(f"输出形状: {embedded.shape}")
print(f"输出:\n{embedded}")
```

这个算子有相当多的作用，例如NLP中基于此创建一个将**离散的 token ID** 映射到**连续向量空间**的查找表。

## TASK 2.3 Linear

公式：

```math
Y = xW^T+b
```

核心代码：

```c++
template <typename T>
void linear_(T *out,
             const T *in,        // [M, K]
             const T *weight,    // [N, K]  (NOTE: not transposed in memory)
             const T *bias_data, // [N] or nullptr
             size_t M,
             size_t K,
             size_t N) {
    // out: [M, N], row-major contiguous
    // Computes: out = in * weight^T + bias

    const bool has_bias = (bias_data != nullptr);

    for (size_t m = 0; m < M; ++m) {
        const T *x_row = in + m * K; // X[m, :]
        T *y_row = out + m * N;      // Y[m, :]

        for (size_t n = 0; n < N; ++n) {
            const T *w_row = weight + n * K; // W[n, :]

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // Accumulate in float for fp16/bf16
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
                }
                if (has_bias) {
                    acc += llaisys::utils::cast<float>(bias_data[n]);
                }
                y_row[n] = llaisys::utils::cast<T>(acc);
            } else {
                // Accumulate in T for normal types (float/double/etc.)
                T acc = T{};
                for (size_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                if (has_bias) {
                    acc += bias_data[n];
                }
                y_row[n] = acc;
            }
        }
    }
}
```

PyTorch示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 方式1: nn.Linear (模块方式)
linear_layer = nn.Linear(10, 5)
x = torch.randn(32, 10)
y1 = linear_layer(x)  # 调用模块

# 方式2: F.linear (函数方式)
weight = torch.randn(5, 10)  # (out_features, in_features)
bias = torch.randn(5)
y2 = F.linear(x, weight, bias)  # 直接调用函数

print(y1.shape, y2.shape)  # 都是 (32, 5)
```

注意，`nn.Linear` 在创建时会**自动创建** `weight` 和 `bias` 参数。

## TASK 2.4 RMS Normalization

给定：

- 输入向量 $x \in \mathbb{R}^d$
- 可学习参数 $\gamma \in \mathbb{R}^d$
- 小常数 $\epsilon$

```math
\text{rms}(x) =
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d} x_i^2
}
```

```math
\hat{x}_i = \frac{x_i}{\text{rms}(x) + \epsilon}
```

$$
y_i = \gamma_i \hat{x}_i
$$

和通常的归一化方法一样，都是为了缩放。

核心代码：

```c++
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
               size_t M, size_t D) {
    for (size_t m = 0; m < M; ++m) {
        const T *x = in + m * D;
        T *y = out + m * D;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // accumulate in float for stability
            float sumsq = 0.0f;
            for (size_t j = 0; j < D; ++j) {
                float v = llaisys::utils::cast<float>(x[j]);
                sumsq += v * v;
            }
            float mean_sq = sumsq / static_cast<float>(D);
            float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

            for (size_t j = 0; j < D; ++j) {
                float xv = llaisys::utils::cast<float>(x[j]);
                float wv = llaisys::utils::cast<float>(weight[j]);
                float outv = (xv * inv_rms) * wv;
                y[j] = llaisys::utils::cast<T>(outv);
            }
        } else {
            double sumsq = 0.0;
            for (size_t j = 0; j < D; ++j) {
                double v = static_cast<double>(x[j]);
                sumsq += v * v;
            }
            double mean_sq = sumsq / static_cast<double>(D);
            double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

            for (size_t j = 0; j < D; ++j) {
                // y = (x * inv_rms) * w
                y[j] = static_cast<T>((static_cast<double>(x[j]) * inv_rms) * static_cast<double>(weight[j]));
            }
        }
    }
}
```

PyTorch示例：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, d)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

# 1. 最基本的调用
dim = 512  # 特征维度
rms_norm = RMSNorm(dim)  # 创建 RMSNorm 实例

# 创建输入数据: (batch_size, dim)
x = torch.randn(32, dim)  # 32个样本，每个512维
output = rms_norm(x)      # 前向传播

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输入前5个值: {x[0, :5]}")
print(f"输出前5个值: {output[0, :5]}")
```

## TASK 2.5 RoPE

著名的旋转位置编码，把位置信息融入到 $Q,K$ 的表示里，并且天然地『相对位置差』形式进入注意力分数。

标准 Self-Attention：

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V
$$

然后：

```math
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```

**RoPE 插在这里：**

```math
Q' = \text{RoPE}(Q)  
```

然后：

```math
\text{Attention}(Q',K',V)
```

**计算方法如下：**

先把向量按两两配对，令每个头的维度为 $d$（通常是偶数），对任意向量 $x\in\mathbb{R}^{d}$ ：

$$
(x_0,x_1),(x_2,x_3),\dots,(x_{d-2},x_{d-1})
$$

把每一对看成二维平面上的一个点，每一对维度有自己的频率，常用定义（也可写作 inv_freq）：

```math
\text{inv\_freq}_i = \theta^{-2i/d}
\quad (i=0,1,\dots,\frac d2-1)
```

位置 $p$ 对应的角度（相位）：

```math
\phi_{p,i} = p \cdot \text{inv\_freq}_i
```

对每一对维度做二维旋转，对第 $i$ 对 $(x_{2i},x_{2i+1})$：

$$
\begin{aligned}
x'_{2i} &= x_{2i}\cos(\phi_{p,i}) - x_{2i+1}\sin(\phi_{p,i})\\
x'_{2i+1} &= x_{2i}\sin(\phi_{p,i}) + x_{2i+1}\cos(\phi_{p,i})
\end{aligned}
$$

这就是二维旋转矩阵：

$$
\begin{pmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{pmatrix}
$$

**RoPE 就是对每个位置 $p$ 的向量做这样一组 block-diagonal 的旋转。**

另外，还有一种配对方式，**隔 half 配对（split-half / NeoX-style）**：$(0,d/2),(1,d/2+1),...$

核心代码：

```c++
template <typename T>
void rope_(T *out,
           const T *in,
           const int64_t *pos_ids,
           float theta,
           size_t seq_len,
           size_t nhead,
           size_t d) {
    const size_t half = d / 2;

    // inv_freq[j] = theta^(-2j/d)
    std::vector<double> inv_freq(half);
    const double log_theta = std::log(static_cast<double>(theta));
    const double inv_d = 1.0 / static_cast<double>(d);
    for (size_t j = 0; j < half; ++j) {
        inv_freq[j] = std::exp(-log_theta * (2.0 * static_cast<double>(j) * inv_d));
    }

    for (size_t t = 0; t < seq_len; ++t) {
        const double pos = static_cast<double>(pos_ids[t]);

        for (size_t h = 0; h < nhead; ++h) {
            const size_t base = (t * nhead + h) * d;
            const T *x = in + base;
            T *y = out + base;

            for (size_t j = 0; j < half; ++j) {
                const double angle = pos * inv_freq[j];
                const double c = std::cos(angle);
                const double s = std::sin(angle);

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    const double a = static_cast<double>(llaisys::utils::cast<float>(x[j]));
                    const double b = static_cast<double>(llaisys::utils::cast<float>(x[j + half]));

                    const double ap = a * c - b * s;
                    const double bp = b * c + a * s;

                    y[j] = llaisys::utils::cast<T>(static_cast<float>(ap));
                    y[j + half] = llaisys::utils::cast<T>(static_cast<float>(bp));
                } else {
                    const double a = static_cast<double>(x[j]);
                    const double b = static_cast<double>(x[j + half]);

                    const double ap = a * c - b * s;
                    const double bp = b * c + a * s;

                    y[j] = static_cast<T>(ap);
                    y[j + half] = static_cast<T>(bp);
                }
            }
        }
    }
}
```

PyTorch示例：

```python
# ----------------------------
# Rotary Embedding (RoPE) - split-half pairing
# Pair dims as (0, D/2), (1, D/2+1), ...
# ----------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        self.dim = dim
        self.max_pos = max_position_embeddings
        self.theta = theta

        # inv_freq[i] = theta^(-2i/dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache [max_pos, dim] for split-half RoPE
        t = torch.arange(self.max_pos, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [max_pos, dim/2]

        # split-half expects cos/sin as [cos, cos] and [sin, sin] along last dim
        emb = torch.cat([freqs, freqs], dim=-1)            # [max_pos, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        q,k: [B, H, T, D]
        positions: [T] or [B,T] (supports KV cache offsets)
        """
        assert q.shape == k.shape, "q and k must have the same shape"
        B, H, T, D = q.shape
        assert D == self.dim, f"Last dim must be {self.dim}, got {D}"

        positions = positions.to(dtype=torch.long, device=q.device)

        # (optional) expand cache if needed
        need = int(positions.max().item()) + 1
        if need > self.cos_cached.size(0):
            self._rebuild_cache(need, device=self.inv_freq.device)

        if positions.dim() == 1:
            # [T] -> [T,D] -> [1,1,T,D]
            cos = self.cos_cached.index_select(0, positions).to(device=q.device, dtype=q.dtype)
            sin = self.sin_cached.index_select(0, positions).to(device=q.device, dtype=q.dtype)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif positions.dim() == 2:
            # [B,T] -> [B,T,D] -> [B,1,T,D]
            flat = positions.reshape(-1)
            cos = self.cos_cached.index_select(0, flat).view(B, T, D).to(device=q.device, dtype=q.dtype)
            sin = self.sin_cached.index_select(0, flat).view(B, T, D).to(device=q.device, dtype=q.dtype)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError(f"positions must be [T] or [B,T], got {positions.shape}")

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k

    @torch.no_grad()
    def _rebuild_cache(self, new_max_pos: int, device=None):
        # rebuild cos/sin cache to length new_max_pos
        self.max_pos = int(new_max_pos)
        dev = device if device is not None else self.inv_freq.device

        t = torch.arange(self.max_pos, dtype=torch.float32, device=dev)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dev))  # [max_pos, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                    # [max_pos, dim]
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    split-half rotate:
    x = [x1, x2] where x1,x2 are each [..., D/2]
    rotate by +90deg in each 2D subspace (x1_i, x2_i):
        (a,b) -> (-b, a)
    returns [-x2, x1]
    """
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)
```

## TASK 2.6 SelfAttention

计算注意力：

```math
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

多头注意力：

设有 $h$ 个头，每个头维度：

$$
d_k = d / h
$$

对每个头：

$$
Q_i = X W_Q^{(i)}
$$

每个头单独做注意力：

```math
\text{head}_i
=
\text{softmax}
\left(
\frac{Q_i K_i^T}{\sqrt{d_k}}
\right)
V_i
```

最后拼接：

```math
\text{MHA}(X)
=
\text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
```

核心代码：

```c++
template <typename T>
void self_attention_(
    T *attn_val,    // [L, nhead, dv]
    const T *query, // [L, nhead, d]
    const T *key,   // [S, nkvhead, d]
    const T *value, // [S, nkvhead, dv]
    size_t L,          // seqlen
    size_t S,          // total_len
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale) {

    if (L <= 0 || S <= 0 || nhead <= 0 || nkvhead <= 0 || d <= 0 || dv <= 0) {
        return;
    }
    if (nhead % nkvhead != 0) {
        return;
    }
    const size_t group = nhead / nkvhead;

    // torch mask: tril(diagonal=S-L)
    // allow keys j <= i + (S - L)
    const size_t shift = S - L; // can be 0 or positive in typical cache case

    for (size_t i = 0; i < L; ++i) { // query position
        size_t j_max = i + shift;    // inclusive
        if (j_max < 0) {
            continue;
        }
        if (j_max >= S) {
            j_max = S - 1;
        }
        const size_t n_keys = j_max + 1;

        for (size_t h = 0; h < nhead; ++h) { // head
            const size_t kvh = h / group;    // repeat_interleave mapping

            const T *q_ptr = query + (static_cast<size_t>(i) * nhead + h) * d;
            T *out_ptr = attn_val + (static_cast<size_t>(i) * nhead + h) * dv;

            // 1) compute logits for allowed keys (masked others are -inf)
            float max_logit = -std::numeric_limits<float>::infinity();
            std::vector<float> exps(static_cast<size_t>(n_keys));

            for (size_t j = 0; j < n_keys; ++j) {
                const T *k_ptr = key + (static_cast<size_t>(j) * nkvhead + kvh) * d;

                float dot = 0.0f;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t t = 0; t < d; ++t) {
                        dot += llaisys::utils::cast<float>(q_ptr[t]) * llaisys::utils::cast<float>(k_ptr[t]);
                    }
                } else {
                    for (size_t t = 0; t < d; ++t) {
                        dot += static_cast<float>(q_ptr[t]) * static_cast<float>(k_ptr[t]);
                    }
                }

                float logit = dot * scale;
                exps[static_cast<size_t>(j)] = logit;
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            // 2) softmax over n_keys
            float sum_exp = 0.0f;
            for (size_t j = 0; j < n_keys; ++j) {
                float e = std::exp(exps[static_cast<size_t>(j)] - max_logit);
                exps[static_cast<size_t>(j)] = e; // reuse buffer to store exp
                sum_exp += e;
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            // 3) weighted sum of V
            // accumulate in float
            std::vector<float> acc(static_cast<size_t>(dv), 0.0f);

            for (size_t j = 0; j < n_keys; ++j) {
                float w = exps[static_cast<size_t>(j)] * inv_sum;
                const T *v_ptr = value + (static_cast<size_t>(j) * nkvhead + kvh) * dv;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t t = 0; t < dv; ++t) {
                        acc[static_cast<size_t>(t)] += w * llaisys::utils::cast<float>(v_ptr[t]);
                    }
                } else {
                    for (size_t t = 0; t < dv; ++t) {
                        acc[static_cast<size_t>(t)] += w * static_cast<float>(v_ptr[t]);
                    }
                }
            }

            // write back
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t t = 0; t < dv; ++t) {
                    out_ptr[t] = llaisys::utils::cast<T>(acc[static_cast<size_t>(t)]);
                }
            } else {
                for (size_t t = 0; t < dv; ++t) {
                    out_ptr[t] = static_cast<T>(acc[static_cast<size_t>(t)]);
                }
            }
        }
    }
}

```

PyTorch实现：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHA)

    Args:
        dim: model dimension C
        num_heads: number of heads H
        dropout: attention dropout prob
        causal: whether to apply causal mask (for decoder / GPT)
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        # One projection for QKV is faster than three separate linears
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x: [B, T, C]
        attn_mask: optional mask broadcastable to [B, 1, T, T] or [B, H, T, T]
                   - can be bool (True=keep, False=mask) or additive (0 / -inf)
        returns:
            y: [B, T, C]
            attn: [B, H, T, T] (attention weights)
        """
        B, T, C = x.shape

        # Project to QKV: [B, T, 3C]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads:
        # q,k,v: [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: [B, H, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Optional causal mask (prevent attending to future tokens)
        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            # mask out positions where causal_mask is False
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Optional user mask
        if attn_mask is not None:
            # If boolean mask: True=keep, False=mask
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                # additive mask: 0 for keep, -inf (or large negative) for mask
                scores = scores + attn_mask

        # Softmax -> attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum: [B, H, T, D]
        y = attn @ v

        # Merge heads: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.out(y)
        y = self.proj_drop(y)

        return y, attn
```

## TASK 2.7 SwiGLU

SwiGLU 是一种 **前馈网络（FFN）里的门控激活函数结构**，SwiGLU 用的是 Swish 激活（也叫 SiLU）：

```math
\text{Swish}(x) = x \cdot \sigma(x)
```

其中：

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

SwiGLU 定义为：

```math
\text{SwiGLU}(x) = (W_a x) \odot \text{Swish}(W_b x)
```

核心代码实现：

```c++
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            const float g = llaisys::utils::cast<float>(gate[i]);
            const float u = llaisys::utils::cast<float>(up[i]);

            // sigmoid(g) = 1 / (1 + exp(-g))
            const float sig = 1.0f / (1.0f + std::exp(-g));
            const float y = u * (g * sig);

            out[i] = llaisys::utils::cast<T>(y);
        } else {
            const T g = gate[i];
            const T u = up[i];

            const T sig = T(1) / (T(1) + static_cast<T>(std::exp(-static_cast<double>(g))));
            out[i] = u * (g * sig);
        }
    }
}
```

PyTorch实现如下：

```python
# ----------------------------
# MLP (SwiGLU)
# ----------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swish(x) = x * sigmoid(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

