# 从零用PyTorch实现Qwen2.5推理

论文：https://arxiv.org/pdf/2412.15115

在本文中，我们要从零使用PyTorch推理Qwen2.5，模型选用：[DeepSeek-R1-Distill-Qwen-1.5B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

具体代码实现会参考Transformers库中的源代码和网上一些其他教程。

## I. 参数与配置

在算法实现之前，我们要实现参数读取和模型配置初始化。

在模型目录中，可以找到 `config.json` ，这个文件记录模型的配置信息，我们在代码中写一个class来读取这个文件

```python
import torch
import json
from typing import Optional, List

class Qwen2Config:
    model_type = "qwen2"

    def __init__(
            self,
            # 词表大小: 模型能识别的唯一字符 / 词元 (token) 总数
            vocab_size: int = 151936,
            # 隐藏层维度: 模型中每个词元的向量表示维度
            hidden_size: int = 1536,
            # 中间层维度: FFN层的隐藏维度
            intermediate_size: int = 8960,
            # 隐藏层数量: Transformer 解码器的层数
            num_hidden_layers: int = 28,
            # 注意力头数量: 多头注意力机制中的头数
            num_attention_heads: int = 12,
            # KV 注意力头数量: 用于分组查询注意力 (GQA)
            num_key_value_heads: int = 2,
            # 最大位置嵌入长度: 模型能处理的最大上下文长度
            max_position_embeddings: int = 131072,  # 修正为配置文件默认值
            # RMSNorm 的 epsilon 值（修正为浮点数类型）
            rms_norm_eps: float = 1e-6,
        
			# 是否绑定词嵌入和输出层权重
            tie_word_embeddings: bool = False,
            # 是否使用 KV 缓存
            use_cache: bool = True,
    		# 是否使用多尺度RoPE
            use_mrope: bool = False,
        	# 是否使用滑动窗口
            use_sliding_window: bool = False,

            # 填充 / 开始 / 结束 token ID
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 151643,
            eos_token_id: int = 151643,

            # RoPE 位置编码参数
            rope_theta: float = 10000.0,
            max_window_layers: int = 21,
        	# 滑动窗口大小
            sliding_window: int = 4096,
            initializer_range: float = 0.02,

            # 注意力层 dropout 率
            attention_dropout: float = 0.0,
            hidden_act: str = "silu",
            transformers_version: str = "4.44.0",
            architectures: Optional[List[str]] = None,

            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # 基础参数赋值
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.use_mrope = use_mrope
        self.use_sliding_window = use_sliding_window
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.max_window_layers = max_window_layers
        self.sliding_window = sliding_window
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.transformers_version = transformers_version
        self.architectures = architectures if architectures else ["Qwen2ForCausalLM"]
        self.torch_dtype = torch_dtype

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典加载配置（核心方法：支持读取配置文件解析后的字典）"""
        # 过滤掉配置文件中存在但类不支持的参数（避免初始化报错）
        valid_keys = cls.__init__.__annotations__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(** filtered_config)

    @classmethod
    def from_json_file(cls, json_file_path: str):
        """从JSON配置文件加载配置（直接读取配置文件的方法）"""
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self):
        """将配置转换为字典（便于保存/验证）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

# 从路径读取配置
def load_model_config(path: str) -> Qwen2Config:
    config_dict = json.load(open(path, "r", encoding="utf-8"))

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }

    if "torch_dtype" in config_dict and isinstance(config_dict["torch_dtype"], str):
        config_dict["torch_dtype"] = dtype_mapping.get(config_dict["torch_dtype"], torch.bfloat16)

    config = Qwen2Config.from_dict(config_dict)

    return config
```

在本文的代码实现中，我们不会做什么通用设计，也不是做训练，而是只针对Qwen2架构进行代码编写进行推理，所以其中**一些参数是用不上**的。

那么下一步我们还要读取模型的权重，在本文的案例中，模型权重以Safetensors格式存放，只要借助safetensors库：

```python
from safetensors.torch import load_file

sd = load_file("./DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors")
```

这个函数返回一个字典，我们可以尝试遍历这个字典的key，看看里面存了什么：

```python
from safetensors.torch import load_file

sd = load_file("./DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors")

for k, v in sd.items():
    print(k, type(v))
```

输出：

```
model.embed_tokens.weight <class 'torch.Tensor'>
model.layers.0.self_attn.q_proj.bias <class 'torch.Tensor'>
model.layers.0.self_attn.k_proj.bias <class 'torch.Tensor'>
model.layers.0.self_attn.v_proj.bias <class 'torch.Tensor'>
model.layers.0.self_attn.q_proj.weight <class 'torch.Tensor'>
model.layers.0.self_attn.k_proj.weight <class 'torch.Tensor'>
model.layers.0.self_attn.v_proj.weight <class 'torch.Tensor'>
model.layers.0.self_attn.o_proj.weight <class 'torch.Tensor'>
model.layers.0.mlp.gate_proj.weight <class 'torch.Tensor'>
model.layers.0.mlp.up_proj.weight <class 'torch.Tensor'>
model.layers.0.mlp.down_proj.weight <class 'torch.Tensor'>
model.layers.0.input_layernorm.weight <class 'torch.Tensor'>
model.layers.0.post_attention_layernorm.weight <class 'torch.Tensor'>
......
model.layers.27.mlp.down_proj.weight <class 'torch.Tensor'>
model.layers.27.input_layernorm.weight <class 'torch.Tensor'>
model.layers.27.post_attention_layernorm.weight <class 'torch.Tensor'>
model.norm.weight <class 'torch.Tensor'>
lm_head.weight <class 'torch.Tensor'>
```

显然，这个文件存放的是模型中每层的参数，数据类型是 torch.Tensor，模型是怎么加载这些参数的呢？后文会提及。

## II. 基本模块实现

现在我们要实现LLM中用到的基本模块，那么具体要实现哪些模块呢？运行以下代码：

```python
from transformers import AutoModelForCausalLM
import torch

model_dir = "./DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    dtype=torch.bfloat16,
    trust_remote_code=True,   # 建议加上
)

print(model)
```

输出：

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```

可知，我们要自己实现Qwen2DecoderLayer，其中又包含了Qwen2Attention和Qwen2MLP以及Qwen2RMSNorm，另外还有Qwen2RotaryEmbedding，接下来要实现这些算法模块。

### Qwen2RMSNorm

这里实现的是RMSNorm，作用是稳定Transformer每层输入的尺度，使深层网络更容易训练。

RMSNorm做的是：
```math
\text{向量} \rightarrow
\frac{\text{向量}}{\text{RMS长度}}
```
再乘一个可学习尺度：
```math
\gamma
```
本质意义：**稳定Transformer每层输入的尺度，使深层网络更容易训练。**

下面来分步给出计算方法以及意义

给定输入向量：
```math
x \in \mathbb{R}^d
```
计算均方值（mean square）：
```math
\text{MS}(x) = \frac{1}{d}\sum_{i=1}^{d} x_i^2
```
代码对应：

```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

这里并不是统计学的 **方差**，而是只是 **平方均值**

计算 RMS（Root Mean Square）：
```math
\text{RMS}(x) = \sqrt{\text{MS}(x)}
```
再归一化
```math
\hat{x}_i =
\frac{x_i}{\sqrt{\text{MS}(x) + \epsilon}}
```
代码对应：

```python
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
```

可学习缩放

最后：
```math
y_i = \gamma_i \hat{x}_i
```
其中：
```math
\gamma \in \mathbb{R}^d
```
代码：

```python
self.weight * hidden_states
```

最后代码实现：

```python
# =========================
# RMSNorm
# =========================

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

### Qwen2MLP

MLP是多层感知机，其作用是对信息做非线性变换，Self-Attention 主要负责：**让 token 之间互相交流信息**

Attention 本身基本是 **线性组合**，真正让模型变得更强表达能力的是 MLP 的非线性计算。

MLP做的事情可以理解为：对每个 token 的特征进行复杂的非线性加工和重编码。

Qwen2 使用 **gated MLP + SiLU(Swish)**，公式
```math
\text{MLP}(x) = \text{down}\big(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)\big)
```
这里要实现一个激活函数：

```python
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
```

代码实现：

```python
# =========================
# MLP (SwiGLU)
# =========================

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
```

其中`nn.Linear`表示一个线性变换：
```math
y = Wx + b
```
这段代码：

```python
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
```

对应的公式：

```math
\text{gate}(x)=W_{gate}x \\
\text{up}(x)=W_{up}x \\
\text{gate}(x)=W_{gate}x
```

### Qwen2Attention

接下来实现最关键的一环，多头注意力计算。

对于输入隐藏状态
```math
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
```
我们通过线性层计算出：
```math
Q = XW_Q + b_Q \\
K = XW_K + b_K \\
V = XW_V + b_V \\
```
我们要对Q和K做旋转位置编码，也就是RoPE，数学上，对位置 $p$ 的向量：
```math
Q_p' = R_p Q_p \\
K_p' = R_p K_p
```
其中 $R_p$ 是由位置 $p$ 决定的分块二维旋转矩阵，具体实现后面会讲到。如果设置了 KV Cache，还要将K和V存到缓存中。

由于我们现在有多个头，注意力的计算结果应该是每个head执行自己的注意力计算，最后汇总起来得到整个注意力分数。

多头注意力的本质思想是：不同头去学习不同类型的关系。

例如某些头更关注：

- 语法依赖
- 长距离指代
- 局部邻近
- 代码括号配对
- 数学符号结构

如果只用一个大头，相当于只做一次注意力：
```math
\text{Attention}(Q,K,V)
```
而多头是并行做很多次：
```math
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
```
再拼接：
```math
\text{MultiHead}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
```
所以，每个小份的维度是原来向量的维度除以头数：

```math
d_h = \frac{d_{\text{model}}}{n_{\text{head}}}
```

在多头的基础上，还用了GQA，每个KV head会服务于多个Q head，也就是说KV head数量要少于Q head的数量，一个KV head服务于一组Q head，组数就是：
```math
g = \frac{n_q}{n_{kv}}
```
那么一对KV就是g个Q head所对应，共同计算出注意力。

整个流程可以概括成：
```math
Q' = \text{RoPE}(Q)
```
然后：
```math
A^{(h)} =
\frac{Q'^{(h)} (K'^{(r(h))})^T}{\sqrt{d_h}}
```
最后，整个计算流程为：
```math
\boxed{
Y
=
\mathrm{Concat}_{h=1}^{n_q}
\left[
\mathrm{Softmax}
\left(
\frac{\mathrm{RoPE}(Q^{(h)})\,\mathrm{RoPE}(K^{(r(h))})^\top}{\sqrt{d_h}} + M
\right)
V^{(r(h))}
\right]
W_O
}
```

#### RoPE

首先实现一下RoPE，即旋转位置编码。

RoPE 的关键思路是：**让 Q 和 K 随位置发生旋转。**

对于位置 $p$，对向量进行旋转：

$$
x_p' = R_p x
$$

其中：

$$
R_p
$$

是旋转矩阵。

RoPE把向量维度两两配对：

$$
(x_{2i},x_{2i+1})
$$

旋转：

```math
\begin{bmatrix}
x'_{2i} \\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i} \\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
```
展开：
```math
x'_{2i} = x_{2i}\cos\theta - x_{2i+1}\sin\theta
```
配对的向量不一定是相邻的，也可以是隔一半长度。

其中旋转角度定义为：
```math
\theta_{p,i} = p \cdot \omega_i
```
其中：
```math
\omega_i = \theta^{-2i/d} \\
\theta = 10000
```
这样的好处是，在计算注意力时有：
```math
\text{Attention}(i,j)=
(Q_i')^T K_j'
```
得到：
```math
Q^T R_{j-i} K
```
可以看到计算只依赖 **相对位置差**

下面看代码实现：

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., head_dim)
    split last dim into two halves:
      x = [x1, x2]
    return [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, n_heads, T, head_dim]
    cos, sin: [B, T, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, T, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
```

在实际代码实现中，我们基于以下公式计算：

```math
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}\times
\begin{bmatrix}
a \\
b
\end{bmatrix}=a\cos\theta-b\sin\theta+a\sin\theta+b\cos\theta=
\begin{bmatrix}
a \\
b
\end{bmatrix}\cos\theta+
\begin{bmatrix}
-b \\
a
\end{bmatrix}\sin\theta
```

将 $x$ 分成 $x_1$ 和 $x_2$ 再拼接成 $[-x_2, x_1]$，带入到上式中就可以算出 $RoPE(x)$ ，基于此我们实现了RoPE函数。

#### Attention

上文提到，一个KV head 服务 多个 Q head，要将数量较少的 Key/Value（KV）头复制扩展，匹配 Query（Q）头的数量，实现 『1 个 KV 头对应 N 个 Q 头』的分组机制：

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

第一步：`[:, :, None, :, :]` 表明插入一个新维度，形状从 `[B, KV_heads, L, D]` 变为 `[B, KV_heads, 1, L, D]`；

第二步：`expand(...)`  沿新插入的维度复制`n_rep`次，形状变为 `[B, KV_heads, n_rep, L, D]`，这个函数不会真的复制新数据，多份副本在底层其实共用一份内存；

最后reshape成期望的形状并返回。

接下来实现注意力计算，我们回顾公式：
```math
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```
按照公式，要将Q与K转置相乘，再进行放缩：

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

代人到 Softmax 计算：

```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

再和V相乘：

```python
attn_output = torch.matmul(attn_weights, value_states)
```

#### 整体实现

基于上文，实现以下代码，即一个注意力层：

```python
# =========================
# Attention
# =========================

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, T, hidden]
        position_embeddings: (cos, sin), each [B, T, head_dim]
        attention_mask: [1 or B, 1, T_q, T_k]
        """
        bsz, q_len, _ = hidden_states.shape

        # Q: [B, T, n_heads * d] -> [B, n_heads, T, d]
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V: [B, T, n_kv_heads * d] -> [B, n_kv_heads, T, d]
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(self.layer_idx, key_states, value_states)

        # GQA: repeat kv heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention scores: [B, n_heads, T_q, T_k]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, n_heads, T_q, d]

        # back to [B, T, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

可以看到，我们的代码中用到了attention mask，是为了在计算prompt中token的注意力分数时，遮盖掉后面的token，这里不做详细解释，详情可以阅读相关论文。

### DecoderLayer

将上面实现的模块封装到解码器中，也就是一个Transformer Block，执行的操作就是：
```math
x \;\longrightarrow\; \text{Norm} \;\longrightarrow\; \text{Attention} \;\longrightarrow\; \text{Residual}
\;\longrightarrow\; \text{Norm} \;\longrightarrow\; \text{MLP} \;\longrightarrow\; \text{Residual}
```
整层公式可以理解成：
```math
h = x + \mathrm{SelfAttn}(\mathrm{RMSNorm}(x)) \\
y = h + \mathrm{MLP}(\mathrm{RMSNorm}(h))
```
我们主要将上面的模块组合到代码中：

```python
# =========================
# Decoder Layer
# =========================

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

### Qwen2RotaryEmbedding

我们额外实现一个模块来生成根据 token 的位置生成旋转角度对应的 cos 和 sin：

```python
# =========================
# Rotary Embedding
# =========================

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x only provides dtype/device.
        position_ids: [B, T]
        returns cos, sin: [B, T, head_dim]
        """
        bsz, seqlen = position_ids.shape
        inv_freq = self.inv_freq[None, :, None].float().expand(bsz, -1, 1).to(x.device)   # [B, d/2, 1]
        pos = position_ids[:, None, :].float()                                              # [B, 1, T]

        freqs = torch.matmul(inv_freq, pos).transpose(1, 2)  # [B, T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)              # [B, T, d]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

核心作用是输出两个张量：

```
cos: [B, T, head_dim]
sin: [B, T, head_dim]
```

之后在 attention 中会这样使用：
```math
q' = q \cdot \cos + \text{rotate}(q) \cdot \sin
```
等价于：
```math
x' = R_\theta x
```
其中 $R_\theta$ 是二维旋转矩阵。

所以这段代码的作用是：**为每个 token、每个维度生成旋转角度的 cos/sin**

### KV Cache

前文提到，我们要缓存每层的QKV，这样避免deocde时期重复计算，这里不做过多赘述。

代码实现如下：

```python
# =========================
# KV Cache
# =========================

class KVCache:
    """
    Per-layer cache:
      keys[layer]   : [B, n_kv_heads, T_cache, head_dim]
      values[layer] : [B, n_kv_heads, T_cache, head_dim]
    """
    def __init__(self, num_layers: int):
        self.keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.values: List[Optional[torch.Tensor]] = [None] * num_layers
	
    # 读写操作合并在一个接口
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        else:
            # 将新的KV追加到后面
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=2)
        return self.keys[layer_idx], self.values[layer_idx] # 返回这层所有KV Cache

    def get_seq_length(self) -> int:
        for k in self.keys:
            if k is not None:
                return k.shape[2]
        return 0
```

## III. 模型实现

下面将上面的模块封装到QwenModel中，先初始化好所有的成员：

```python
class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) # embedding层
        
        # 堆叠Decoder层
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 归一化层
        self.rotary_emb = Qwen2RotaryEmbedding(config) # RoPE层
```

接下来实现forward函数，先理清前向计算的流程。

模型接收Token ID序列，首先用Embedding将一串Token转化成词向量。

比如一句话经过 tokenizer 后，被转化成一个Token ID序列。

```
input_ids = [[101, 2054, 2003, 102]]
```

Embedding会把每个ID映射成一个向量：

```python
inputs_embeds = self.embed_tokens(input_ids)  # [B, T, hidden]
```

将这堆向量再输入到RoPE进行编码：

```python
position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
```

构造 causal mask：

```python
# causal mask
kv_len = past_seen_tokens + seqlen
attention_mask = make_causal_mask(seqlen, kv_len, device=device, dtype=inputs_embeds.dtype)
```

接着经过多层 decoder 进行计算：

```python
hidden_states = inputs_embeds
for layer in self.layers:
    hidden_states = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values if use_cache else None,
    )
```

所有 decoder layer 跑完以后，再做一次 RMSNorm：

```python
hidden_states = self.norm(hidden_states)
```

这就是我们要的结果了，最终的 `hidden_states` 是每个 token 在『整个上下文条件下』的语义表示向量。

融合了：

1. token 本身语义
2. token 的位置信息（RoPE）
3. 与前面所有 token 的上下文关系（attention）
4. 多层 Transformer 的非线性语义抽象

代码实现如下：

```python
# =========================
# Base Model
# =========================

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        input_ids:
          prefill: [B, T]
          decode : [B, 1]
        """
        device = input_ids.device
        bsz, seqlen = input_ids.shape

        inputs_embeds = self.embed_tokens(input_ids)  # [B, T, hidden]

        # cache positions
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        position_ids = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seqlen,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)  # [B, T]

        # RoPE table
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # causal mask
        kv_len = past_seen_tokens + seqlen
        attention_mask = make_causal_mask(seqlen, kv_len, device=device, dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values if use_cache else None,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values
```

现在到了最后一步，完成模型的文本生成功能。

QwenModel输出的hidden_states还要通过一个线性层，**将其投影成整个词表的 logits，用来预测下一个 token**，这就是**lm_head**层。

这个logits就是每个词的概率分布，可以通过各种采样方式，得到最后的next token，也就是这轮生成的token了。

完整代码实现：

```python
class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)  # [B, T, vocab]
        return logits, past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy decoding.
        """
        self.eval()
        device = input_ids.device

        # 1) prefill
        cache = KVCache(num_layers=self.config.num_hidden_layers)
        logits, cache = self.forward(input_ids, past_key_values=cache, use_cache=True)

        generated = input_ids
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]
        generated = torch.cat([generated, next_token], dim=1)

        # 2) decode one token at a time
        for _ in range(max_new_tokens - 1):
            logits, cache = self.forward(next_token, past_key_values=cache, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                if torch.all(next_token == eos_token_id):
                    break

        return generated
```

在generate函数中，生成过程分成两个阶段，即Prefill和Docode

假设输入的是[A, B, C]，生成的是[D, E, F]，那么生成D的过程就是做Prefill，接着一次Decode生成E，一次Decode生成F

在Decode阶段，在每轮生成中，因为有KV Cache所以不用把所有已生成的token输入到模型，只用把token输入到模型中，用argmax采样得到生成的token，拼接到已生成的序列中，当达到最大token数或者遇到终止符号EOS，则停止生成。

## IV. 调用模型

用如下代码调用这个模型：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "./DeepSeek-R1-Distill-Qwen-1.5B" # 模型路径

tokenizer = AutoTokenizer.from_pretrained(model_dir) # 创建tokenize
config = load_model_config(model_dir + "/config.json")
config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

model = Qwen2ForCausalLM(config).to(device)
model.load_state_dict(load_file(model_dir + "/model.safetensors"))

start = "北京大学是" # 提示词
start_ids = tokenizer(start).data['input_ids']
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=1024)
res = tokenizer.decode(y[0].tolist())  # 解码生成的 token 序列为可读文本

print(res)
```

输出：

```
<｜begin▁of▁sentence｜>北京大学是国家的象征，同时也是国际社会的权威机构，它在国际事务中扮演着重要角色。北京大学的教育体系培养了大量人才，这些人才在国际舞台上取得了显著成就。因此，北京大学的教育体系对国家的发展至关重要。....
```

整个推理过程的**完整代码**：

```python
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file

import json
from transformers import AutoTokenizer

# =========================
# Config
# =========================

class Qwen2Config:
    model_type = "qwen2"

    def __init__(
            self,
            # 词表大小: 模型能识别的唯一字符 / 词元 (token) 总数
            vocab_size: int = 151936,
            # 隐藏层维度: 模型中每个词元的向量表示维度
            hidden_size: int = 1536,
            # 中间层维度: FFN层的隐藏维度
            intermediate_size: int = 8960,
            # 隐藏层数量: Transformer 解码器的层数
            num_hidden_layers: int = 28,
            # 注意力头数量: 多头注意力机制中的头数
            num_attention_heads: int = 12,
            # KV 注意力头数量: 用于分组查询注意力 (GQA)
            num_key_value_heads: int = 2,
            # 最大位置嵌入长度: 模型能处理的最大上下文长度
            max_position_embeddings: int = 131072,  # 修正为配置文件默认值
            # RMSNorm 的 epsilon 值（修正为浮点数类型）
            rms_norm_eps: float = 1e-6,

            tie_word_embeddings: bool = False,
            # 是否使用 KV 缓存
            use_cache: bool = True,
            use_mrope: bool = False,
            use_sliding_window: bool = False,

            # 填充 / 开始 / 结束 token ID
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 151643,
            eos_token_id: int = 151643,

            # RoPE 位置编码参数
            rope_theta: float = 10000.0,
            max_window_layers: int = 21,
            sliding_window: int = 4096,
            initializer_range: float = 0.02,

            # 注意力层 dropout 率
            attention_dropout: float = 0.0,
            hidden_act: str = "silu",
            transformers_version: str = "4.44.0",
            architectures: Optional[List[str]] = None,

            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # 基础参数赋值
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.use_mrope = use_mrope
        self.use_sliding_window = use_sliding_window
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.max_window_layers = max_window_layers
        self.sliding_window = sliding_window
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.transformers_version = transformers_version
        self.architectures = architectures if architectures else ["Qwen2ForCausalLM"]
        self.torch_dtype = torch_dtype
        self.layer_types = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典加载配置（核心方法：支持读取配置文件解析后的字典）"""
        # 过滤掉配置文件中存在但类不支持的参数（避免初始化报错）
        valid_keys = cls.__init__.__annotations__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(** filtered_config)

    @classmethod
    def from_json_file(cls, json_file_path: str):
        """从JSON配置文件加载配置（直接读取配置文件的方法）"""
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self):
        """将配置转换为字典（便于保存/验证）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def load_model_config(path: str) -> Qwen2Config:
    config_dict = json.load(open(path, "r", encoding="utf-8"))

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }

    if "torch_dtype" in config_dict and isinstance(config_dict["torch_dtype"], str):
        config_dict["torch_dtype"] = dtype_mapping.get(config_dict["torch_dtype"], torch.bfloat16)

    config = Qwen2Config.from_dict(config_dict)

    return config


# =========================
# KV Cache
# =========================

class KVCache:
    """
    Per-layer cache:
      keys[layer]   : [B, n_kv_heads, T_cache, head_dim]
      values[layer] : [B, n_kv_heads, T_cache, head_dim]
    """
    def __init__(self, num_layers: int):
        self.keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.values: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self) -> int:
        for k in self.keys:
            if k is not None:
                return k.shape[2]
        return 0

# =========================
# Utils
# =========================

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., head_dim)
    split last dim into two halves:
      x = [x1, x2]
    return [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, n_heads, T, head_dim]
    cos, sin: [B, T, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, T, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    [B, n_kv_heads, T, d] -> [B, n_heads, T, d]
    """
    bsz, n_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, n_kv_heads, n_rep, seqlen, head_dim
    )
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, seqlen, head_dim)


def make_causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return shape: [1, 1, q_len, kv_len]
    Mask future positions with -inf.
    Works for both prefill and decode.
    """
    # allow query position i to attend to key position <= current absolute position
    # In incremental decoding with cache, q_len is often 1, kv_len grows.
    mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1 + (kv_len - q_len))
    return mask.unsqueeze(0).unsqueeze(0).to(dtype)


# =========================
# RMSNorm
# =========================

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# =========================
# MLP (SwiGLU)
# =========================

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


# =========================
# Rotary Embedding
# =========================

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x only provides dtype/device.
        position_ids: [B, T]
        returns cos, sin: [B, T, head_dim]
        """
        bsz, seqlen = position_ids.shape
        inv_freq = self.inv_freq[None, :, None].float().expand(bsz, -1, 1).to(x.device)   # [B, d/2, 1]
        pos = position_ids[:, None, :].float()                                              # [B, 1, T]

        freqs = torch.matmul(inv_freq, pos).transpose(1, 2)  # [B, T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)              # [B, T, d]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# =========================
# Attention
# =========================

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, T, hidden]
        position_embeddings: (cos, sin), each [B, T, head_dim]
        attention_mask: [1 or B, 1, T_q, T_k]
        """
        bsz, q_len, _ = hidden_states.shape

        # Q: [B, T, n_heads * d] -> [B, n_heads, T, d]
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V: [B, T, n_kv_heads * d] -> [B, n_kv_heads, T, d]
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(self.layer_idx, key_states, value_states)

        # GQA: repeat kv heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention scores: [B, n_heads, T_q, T_k]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, n_heads, T_q, d]

        # back to [B, T, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# =========================
# Decoder Layer
# =========================

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =========================
# Base Model
# =========================

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        input_ids:
          prefill: [B, T]
          decode : [B, 1]
        """
        device = input_ids.device
        bsz, seqlen = input_ids.shape

        inputs_embeds = self.embed_tokens(input_ids)  # [B, T, hidden]

        # cache positions
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        position_ids = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seqlen,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)  # [B, T]

        # RoPE table
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # causal mask
        kv_len = past_seen_tokens + seqlen
        attention_mask = make_causal_mask(seqlen, kv_len, device=device, dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values if use_cache else None,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


# =========================
# Causal LM
# =========================

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # tie weights
        # self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)  # [B, T, vocab]
        return logits, past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy decoding.
        """
        self.eval()
        device = input_ids.device

        # 1) prefill
        cache = KVCache(num_layers=self.config.num_hidden_layers)
        logits, cache = self.forward(input_ids, past_key_values=cache, use_cache=True)

        generated = input_ids
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]
        generated = torch.cat([generated, next_token], dim=1)

        # 2) decode one token at a time
        for _ in range(max_new_tokens - 1):
            logits, cache = self.forward(next_token, past_key_values=cache, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                if torch.all(next_token == eos_token_id):
                    break

        return generated


device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "./DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = load_model_config(model_dir + "/config.json")
config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

model = Qwen2ForCausalLM(config).to(device)
model.load_state_dict(load_file(model_dir + "/model.safetensors"))

start = ""
start_ids = tokenizer(start).data['input_ids']
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=1024)
res = tokenizer.decode(y[0].tolist())  # 解码生成的 token 序列为可读文本

print(res)

```

`
