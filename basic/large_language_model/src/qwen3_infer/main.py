import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file

import json
from transformers import AutoTokenizer
import os

# =========================
# Config
# =========================

class Qwen3Config:
    model_type = "qwen3"

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

            head_dim: int = 128,

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
        self.head_dim = head_dim
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
        self.architectures = architectures if architectures else ["Qwen3ForCausalLM"]
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


def load_model_config(path: str) -> Qwen3Config:
    config_dict = json.load(open(path, "r", encoding="utf-8"))

    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }

    if "torch_dtype" in config_dict and isinstance(config_dict["torch_dtype"], str):
        config_dict["torch_dtype"] = dtype_mapping.get(config_dict["torch_dtype"], torch.bfloat16)

    config = Qwen3Config.from_dict(config_dict)

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

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# =========================
# MLP (SwiGLU)
# =========================

class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
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

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        head_dim = config.head_dim
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

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

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
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Q: [B, T, n_heads * d] -> [B, n_heads, T, d]
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        # K/V: [B, T, n_kv_heads * d] -> [B, n_kv_heads, T, d]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

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

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
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


def load_sharded_safetensors(model_dir: str):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    for shard_name in shard_files:
        shard_path = os.path.join(model_dir, shard_name)
        shard_state = load_file(shard_path)

        overlap = set(state_dict).intersection(shard_state)
        if overlap:
            raise ValueError(f"发现重复参数键: {overlap}")

        state_dict.update(shard_state)

    return state_dict



device = "cuda"
model_dir = "./Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = load_model_config(model_dir + "/config.json")
config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

model = Qwen3ForCausalLM(config).to(device)
model.load_state_dict(load_sharded_safetensors(model_dir))

start = "杭州电子科技大学是"
start_ids = tokenizer(start).data['input_ids']
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=1024)
res = tokenizer.decode(y[0].tolist())  # 解码生成的 token 序列为可读文本

print(res)

