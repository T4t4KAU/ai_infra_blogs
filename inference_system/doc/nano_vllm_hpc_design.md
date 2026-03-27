# Nano VLLM 高性能设计

Nano-vLLM 是一个轻量级的大语言模型推理引擎，旨在提供与 vLLM 相当的推理速度，同时保持代码的可读性和简洁性。

链接：https://github.com/GeeeekExplorer/nano-vllm

本文主要讲解这个框架的高性能设计。

## 前缀缓存

这是一项重要的优化技术，通过识别和重用重复的输入前缀，避免了重复的计算和存储。

这项技术实现在BlockManager中，使用xxhash算法计算 token 序列，支持基于前缀哈希的链式计算，确保前缀产生相同哈希值。

在为序列分配缓存块时，先计算块的哈希值，从哈希表中检索是否有已经映射的 block，如果已有就直接复用，其他情况都要新开辟。

```python
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1) # 查哈希表
        
        # 缓存未命中
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        # 缓存命中
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id] # 复用缓存块
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

与之对应的释放操作：

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

释放一个序列时，要将其占用或者引用的 block 做处理。

## CUDA Graph

CUDA 图优化是一种将多次 CUDA 操作捕获为单个图的技术，减少了内核启动开销，提高了推理速度。

CUDA Graph 本质上是把一串原本需要 CPU 一次次提交的 GPU 操作，先录制为一个带依赖关系的有向无环图（DAG），然后以后直接整图重放。这样做的核心收益，不是让单个 kernel 更快，而是把大量 kernel launch 的 CPU 开销压缩掉，用一次图启动代替许多次单独提交。

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    
    # 创建静态张量
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        
        # 为每个 batch_size 创建 graph
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        
        torch.cuda.synchronize() # 确保当前 batch 的 warmup / capture 相关 GPU 工作都彻底完成
        reset_context()
        
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

这段代码提前为多个 batch size 录制 CUDA Graph，后面推理时就可以直接 replay，减少 Python/CPU 发射 kernel 的开销。

它不是执行真实推理请求，而是提前构造一批静态推理模板。后面如果来了 batch size = 1、2、4、8、16、32 ... 的请求，就可以直接复用对应的 graph，下面有一个 warmup 过程，避免触发 lazy initialization，让 cudnn/cublas/allocator 等完成第一次初始化，避免把一次性的初始化行为录进 graph，稳定后续 capture 的 kernel 序列。

如下是调用：

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    # 排除不使用 graph 的情况
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] # 获取 bs 对应的 graph
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

## 张量并行

在 LLM Engine 的 初始化代码中：

```python
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event() # 创建一个事件对象，用于多进程同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建一个子进程
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 为主进程创建0号ModelRunner
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

主要做了：

- 在 LLMEngine 中，为每个 GPU 创建一个 ModelRunner 进程
- 使用 PyTorch 的分布式训练功能初始化进程组
- 每个进程负责模型的一部分计算



在模型加载时，根据张量并行大小分割模型权重，每个 GPU 负责处理模型的一部分层或特征维度，通过分布式通信同步必要的中间结果。代码如下：

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info() # 查看显存信息
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 分割 KV 头
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

根据当前 GPU 剩余可用显存，估算能分配多少个 KV cache block，然后一次性申请整块 KV cache，并把每一层 attention 模块的 `k_cache` / `v_cache` 指向这块大缓存的对应切片。

