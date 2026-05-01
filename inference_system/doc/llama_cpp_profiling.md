# Llama.cpp 性能分析方法

llama.cpp 是一个用 C++ 编写的端侧推理系统，面向的是资源受限的平台。本文将尝试用一些系统分析工具对此进行性能分析。

## Mental Model

一次 LLM 推理通常可以拆成以下阶段：

| 阶段 | 典型行为 | 常见指标 | 常见瓶颈 |
| :-: | :-: | :-: | :-: |
| 模型加载 | mmap 或读取 GGUF，初始化 backend，分配 KV cache | load time、page faults、VRAM/RAM 占用 | 磁盘、page cache、内存不足、offload 布局 |
| warmup | 触发一次小图计算，让权重、kernel、CUDA graph 进入稳定状态 | 首次运行耗时 | JIT、lazy init、CUDA graph warmup |
| prompt processing / prefill | 一次处理很多 prompt token，填充 KV cache | prompt eval t/s、TTFT | batch/ubatch、矩阵乘、attention、内存带宽 |
| token generation / decode | 每次通常处理 1 个新 token，循环采样 | eval t/s、TPOT、latency | 小 batch kernel launch、KV 读写、CPU 同步、采样 |
| sampling / detokenization | logits 后处理、采样、输出文本 | sample time、CPU flame graph | grammar、top-k/top-p、JSON/schema、字符串处理 |
| server batching | 多请求 slot 组 batch，共享一个 `llama_context` | QPS、p50/p95、busy slots、KV usage | 队列、slot 复用、连续 batching、上下文碎片 |

## Profiling Build

### CPU profiling build

采用 CPU-only 的方式，执行如下命令，构建用于 profiling 的产物：

```bash
cmake -B build-cpu-prof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_NATIVE=ON \ # 启用 本机 CPU 指令集优化
  -DGGML_OPENMP=ON \ # 开启 OpenMP 多线程并行
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \ # 保留调用栈
  -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"

cmake --build build-cpu-prof --config RelWithDebInfo -j
```

### CUDA profiling build

```bash
cmake -B build-cuda-prof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \ # 带调试信息的发布版本
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_CUDA_FLAGS="-lineinfo"

cmake --build build-cuda-prof --config RelWithDebInfo -j
```

若要对比 CUDA Graphs，可再构建一个关闭 CUDA Graphs 的目录：

```bash
cmake -B build-cuda-nograph-prof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build-cuda-nograph-prof --config RelWithDebInfo -j
```

执行这个命令可以验证：

```bash
./build-cuda-prof/bin/llama-bench --list-devices
./build-cuda-prof/bin/llama-cli --version
```

## 建立可重复 baseline

先创建输出目录：

```bash
cd llama.cpp
export MODEL=/path/to/Qwen3-8B-f16.gguf
mkdir -p profiles/{bench,nsys,ncu,perf,server,logs}
```

保存环境快照：

```bash
{
  date -Iseconds
  git rev-parse HEAD
  ./build/bin/llama-cli --version
  cmake -LAH build | rg '^(CMAKE_BUILD_TYPE|GGML_|LLAMA_BUILD_)'
  nvidia-smi
  nvcc --version
  nsys --version
  ncu --version
  perf --version
  lscpu
  free -h
} | tee profiles/logs/env-$(date +%Y%m%d-%H%M%S).txt
```

### baseline 的原则

- 每次只改一个变量。例如只改 `-t`，不要同时改 `-t`、`-b`、`-fa`、`-ngl`
- 分开看 prefill 和 decode：`-p N -n 0` 看 prefill，`-p 0 -n N` 看 decode
- 使用 `-r 3` 或更多重复次数看波动。正式数字用 `-r 5` 或 `-r 10`
- profiler 下只跑短测试。长测试先用 `llama-bench` 找问题，再用 nsys/ncu 抓一小段
- 记录失败也有价值，尤其是 OOM、KV slot 不足、GPU offload 失败

### CPU baseline

f16 8B 在 CPU 上会慢，先用较小 token 数确认流程

```bash
./build/bin/llama-bench \
  -m "$MODEL" \
  -ngl 0 \ # GPU layer 数量
  -p 128 \
  -n 16 \
  -t 1,2,4,8,14,20 \ # 指定线程数量
  -r 3 \ # 指定重复次数
  -o jsonl \
  | tee profiles/bench/cpu_threads_qwen3_f16.jsonl
```

解读：

- 如果 `-t 20` 反而慢，说明线程过多造成调度或内存带宽竞争
- i5-14600KF 有 P-core/E-core 混合结构，`-t` 最优值不一定等于 logical CPU 数
- decode 阶段常常比 prefill 更容易被 CPU 同步、采样、内存访问影响

### CUDA baseline

由于 f16 模型约 16 GB，像 RTX 5070 12 GB 这样的显卡无法完整吃下，先用较小 `-ngl`：

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 0,8,16,24 \
  -fa 0,1 \
  -p 512 \
  -n 64 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/cuda_ngl_fa_qwen3_f16.jsonl
```

如果 OOM：

- 降低 `-ngl`
- 降低 `-c` 或 benchmark 的 context depth
- 尝试 `-ctk q8_0 -ctv q8_0` 或更小 KV 类型，并验证兼容性
- 使用量化模型做 GPU 完整 offload 对照，例如 Q4_K_M 或 Q8_0

### 记录吞吐和稳定性

`llama-bench -o jsonl` 的字段包括 `avg_ts`、`stddev_ts`、`avg_ns`、`stddev_ns`，观察：

- `avg_ts`：平均 tokens/s。
- `stddev_ts / avg_ts`：相对波动。波动大可能是温度、频率、后台进程、page faults 或 profiler 干扰
- `n_prompt`、`n_gen`、`n_depth`：确认你测的是 prefill、decode 还是带上下文深度的 decode

如果安装了 `jq`，可以提取关键字段：

```bash
jq -r '[.backend,.n_gpu_layers,.flash_attn,.n_prompt,.n_gen,.n_depth,.n_threads,.avg_ts,.stddev_ts] | @tsv' \
  profiles/bench/cuda_ngl_fa_qwen3_f16.jsonl
```

## 用内部计时先分段

`llama.cpp` 内部有上下文计时，主要从 `llama_perf_context_print()` 打印：

- load time
- prompt eval time
- eval time
- total time
- graphs reused

CLI 例子：

```bash
./build-cuda-prof/bin/llama-cli \
  -m "$MODEL" \
  -p "用三句话解释推理系统 profiling 的目标" \
  -n 64 \
  -ngl 16 \
  -fa on \
  --perf \
  --no-warmup \
  --seed 42 \
  2>&1 | tee profiles/logs/cli_cuda_perf.log
```

建议先跑两次：

```bash
# 第一次包含更多冷启动因素
./build-cuda-prof/bin/llama-cli -m "$MODEL" -p "hello" -n 16 -ngl 16 --perf

# 第二次更接近 page cache / backend 初始化后的状态
./build-cuda-prof/bin/llama-cli -m "$MODEL" -p "hello" -n 16 -ngl 16 --perf
```

观察方法：

- `load time` 很大：先查 mmap、磁盘、page cache、`--mlock`、模型大小
- `prompt eval time` 很大：看 batch/ubatch、GPU offload、BLAS/CUDA、Flash Attention
- `eval time` 很大：看 decode 循环、KV cache、CPU 同步、kernel launch 间隔
- `graphs reused` 低：CUDA graphs 可能没有稳定复用，或每轮图形状变化太大

`llama-bench` 的说明中明确提到它的测量不包含 tokenization 和 sampling 时间，CLI/server 的端到端表现会多出这些部分

## 用 llama-bench 做参数扫描

`llama-bench` 是最适合建立性能曲线的工具。它的关键能力是一个参数给多个值，自动跑组合

### prefill batch 扫描

目标：看 prompt processing 是否吃满矩阵乘和 GPU/CPU

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 16 \
  -fa 1 \
  -n 0 \
  -p 256,512,1024 \
  -b 256,512,1024,2048 \
  -ub 128,256,512 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/prefill_batch_ubatch.jsonl
```

看点：

- `n_batch` 增大后 prefill t/s 是否提升
- `n_ubatch` 增大后是否 OOM，或因为显存压力下降速
- 小 prompt 时吞吐低不一定是问题，可能是固定开销占比高

### decode 长度扫描

目标：看生成阶段是否稳定，是否随生成长度下降

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 16 \
  -fa 1 \
  -p 0 \
  -n 16,64,128,256 \
  -r 5 \
  -o jsonl \
  | tee profiles/bench/decode_len.jsonl
```

看点：

- `tg16` 和 `tg256` 差异大：短 decode 被固定开销、warmup、计时噪声影响更明显
- `tg` 随长度变慢：可能是上下文深度、KV cache、温度/功耗、显存换页或 CPU 同步问题

### 上下文深度扫描

目标：观察 decode 随 KV cache 长度增长的成本

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 16 \
  -fa 1 \
  -p 0 \
  -n 64 \
  -d 0,512,1024,2048,4096 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/decode_depth.jsonl
```

看点：

- `tg64 @ d4096` 比 `tg64 @ d0` 慢多少
- Flash Attention 对长上下文是否改善
- KV cache 类型是否改变显存占用和速度

### GPU offload 扫描

找出 12 GB VRAM 下的最高可用 offload 层数，以及性能拐点

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 0,4,8,12,16,20,24 \
  -fa 1 \
  -p 512 \
  -n 64 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/ngl_sweep.jsonl
```

看点：

- 性能是否随 `-ngl` 单调提高。若不是，可能是 CPU/GPU 之间拷贝、显存压力或调度开销
- 某个 `-ngl` 开始 OOM，记录这个边界
- partial offload 下，Nsight Systems 特别要看 H2D/D2H 拷贝和 CPU/GPU 空洞

### Flash Attention 和 KV 类型扫描

目标：看 attention 实现和 KV cache 精度对长上下文的影响。

```bash
./build-cuda-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 16 \
  -fa 0,1 \
  -ctk f16,q8_0 \
  -ctv f16,q8_0 \
  -p 512 \
  -n 64 \
  -d 2048 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/fa_kvtype.jsonl
```

如果某些组合报错，说明当前模型、backend 或 attention 路径不支持该组合。不要把报错组合当成性能结论。

## 用 Nsight Systems 看系统时间线

Nsight Systems 适合回答“时间去哪了”。它看的是跨 CPU 线程、CUDA API、GPU kernel、memcpy、OS runtime 的时间线

### 第一次抓取：短 benchmark

先用短命令，避免生成巨大报告。

```bash
nsys profile \
  --force-overwrite=true \
  --output=profiles/nsys/qwen3_pp512_tg64_ngl16 \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --backtrace=fp \
  --cuda-graph-trace=graph \
  ./build-cuda-prof/bin/llama-bench \
    -m "$MODEL" \
    -ngl 16 \
    -fa 1 \
    -p 512 \
    -n 64 \
    -r 1
```

生成的 `.nsys-rep` 可以用 GUI 打开：

```bash
nsys-ui profiles/nsys/qwen3_pp512_tg64_ngl16.nsys-rep
```

也可以先用 CLI 摘要：

```bash
nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum \
  --format table \
  profiles/nsys/qwen3_pp512_tg64_ngl16.nsys-rep \
  | tee profiles/nsys/qwen3_pp512_tg64_ngl16_stats.txt
```

常用报告：

| report                  | 用途                                    |
| :---------------------- | :-------------------------------------- |
| `cuda_gpu_kern_sum`     | 哪些 kernel 占 GPU 时间                 |
| `cuda_kern_exec_sum`    | kernel launch 到执行的延迟和执行时间    |
| `cuda_api_sum`          | CPU 侧 CUDA API 花费，尤其同步和 memcpy |
| `cuda_gpu_mem_time_sum` | H2D、D2H、D2D 拷贝耗时                  |
| `osrt_sum`              | mutex、pthread、文件 IO 等 OS runtime   |
| `nvtx_sum`              | 如果你自己加了 NVTX，可按 range 汇总    |

### 在 GUI 里看什么

按这个顺序看：

1. Timeline 顶部总览：GPU 是否大段空闲
2. CUDA API lane：是否大量 `cudaStreamSynchronize`、`cudaMemcpyAsync`、`cudaGraphLaunch`
3. GPU lane：kernel 是否密集，kernel 之间是否有大间隔
4. CPU sampling：热栈是否在 `llama_decode`、`ggml_backend_sched_graph_compute_async`、采样、日志或锁
5. Memcpy：生成阶段是否反复 D2H/H2D
6. CUDA Graph：是否从许多小 kernel launch 变成 `cudaGraphLaunch`，以及 graph 是否稳定复用

典型判断：

- GPU kernel 很短且间隔很大：可能是 launch overhead、CPU 调度、CUDA graph 没复用、batch 太小

- GPU 长时间 0% busy，但 CPU 某线程满：CPU 成为瓶颈，查 `perf`

- 每个 token 都有大 D2H：可能在取 logits、采样或 CPU/GPU 边界搬运

  prefill GPU 很忙，decode GPU 不忙：正常但仍可优化，如 batching、CUDA graphs、server 连续 batching。

- partial offload 下 CPU 和 GPU 交替：可能 `-ngl` 太低或层切分导致跨设备传输

### 对比 CUDA Graphs

用两个 build 对比：

```bash
# CUDA graphs 开启
nsys profile --force-overwrite=true \
  -o profiles/nsys/graphs_on \
  --trace=cuda,nvtx,osrt,cublas \
  ./build-cuda-prof/bin/llama-bench -m "$MODEL" -ngl 16 -fa 1 -p 0 -n 128 -r 1

# CUDA graphs 关闭
nsys profile --force-overwrite=true \
  -o profiles/nsys/graphs_off \
  --trace=cuda,nvtx,osrt,cublas \
  ./build-cuda-nograph-prof/bin/llama-bench -m "$MODEL" -ngl 16 -fa 1 -p 0 -n 128 -r 1
```

看点：

- `graphs_on` 是否出现 `cudaGraphLaunch`
- kernel launch API 数量是否下降
- `llama_perf_context_print` 中 `graphs reused` 是否增加
- decode t/s 是否提升

### 不要过早打开所有开关

下面这些选项很有用，但开销大。只有在需要时打开：

```bash
--cuda-memory-usage=true
--cuda-event-trace=true
--cuda-trace-all-apis=true
--cudabacktrace=kernel:50000,memory:50000,sync:50000
```

经验：第一轮只抓 `cuda,nvtx,osrt,cublas`，发现疑点后再打开重型选项。

## 用 Nsight Compute 深挖 CUDA kernel

Nsight Compute 适合回答“某个 kernel 为什么慢”。它不适合第一次上来就 profile 整个推理，因为它会 replay kernel，开销很大。

推荐流程：

1. 先用 Nsight Systems 找到热 kernel 名称。
2. 用 `ncu --kernel-name` 只抓少数 kernel。
3. 先用 `--set basic`，再逐步加 section。

示例：抓第 30 次之后的 5 个匹配 kernel。

```bash
ncu \
  --set basic \
  --target-processes all \
  --launch-skip 30 \
  --launch-count 5 \
  --kernel-name "regex:.*(mul|gemm|mmq|mmv|fattn|flash).*" \
  --export profiles/ncu/qwen3_hotkernels \
  --force-overwrite \
  ./build-cuda-prof/bin/llama-bench \
    -m "$MODEL" \
    -ngl 16 \
    -fa 1 \
    -p 0 \
    -n 64 \
    -r 1
```

打开 GUI：

```bash
ncu-ui profiles/ncu/qwen3_hotkernels.ncu-rep
```

或直接看 CLI 输出：

```bash
ncu --set basic --launch-skip 30 --launch-count 3 \
  ./build-cuda-prof/bin/llama-bench -m "$MODEL" -ngl 16 -fa 1 -p 0 -n 16 -r 1
```

### 关键指标怎么读

| 指标                           | 说明                 | 常见解释                                                   |
| :----------------------------- | :------------------- | :--------------------------------------------------------- |
| SM Throughput                  | SM 计算繁忙程度      | 低可能是内存、launch、依赖或 occupancy 问题                |
| Memory Throughput              | 显存带宽利用         | 高且 SM 低，偏 memory-bound                                |
| Achieved Occupancy             | 实际 occupancy       | 低不一定坏，Tensor Core kernel 也可能 occupancy 不高但很快 |
| Warp Stall Reasons             | warp 等待原因        | 等内存、同步、指令依赖                                     |
| Tensor Core / pipe utilization | 是否用了矩阵加速路径 | f16/bf16 GEMM 应关注                                       |
| L2 hit rate                    | cache 命中           | KV cache 和权重访问可能受影响                              |

### 针对 llama.cpp 的 kernel 关注点

- 矩阵乘：`mul_mat`、`mmq`、`mmv`、cuBLAS GEMM。prefill 大多看这里。
- attention：`fattn`、softmax、rope、mask。长上下文 decode 常看这里。
- norm/activation：RMSNorm、SiLU、元素级 kernel。单个不大，但数量多会造成 launch overhead。
- memcpy：如果 nsys 显示 memcpy 热，ncu 不是第一工具，先回到数据放置和 logits/KV offload。

### ncu 常见坑

- ncu 会显著改变运行时间，不要用它的总耗时当吞吐结论
- replay 需要 kernel 可重放。复杂 server workload 可能不稳定，优先用 `llama-bench`
- CUDA Graph 下 kernel 命名和采集方式可能不同，可试 `--graph-profiling node` 或临时关闭 CUDA Graphs
- 如果报告太大，减少 `--launch-count`

## 用 perf 分析 CPU 路径

CPU-only、partial offload、sampling、server JSON 处理、线程调度都离不开 CPU profiler。

### perf stat：先看宏观计数

```bash
perf stat -d -r 3 -- \
  ./build-cpu-prof/bin/llama-bench \
    -m "$MODEL" \
    -ngl 0 \
    -p 128 \
    -n 16 \
    -t 14 \
    -r 1
```

关注：

- `cycles`、`instructions`、`IPC`：CPU 是否有效执行。
- `cache-misses`：内存访问压力。
- `context-switches`：线程调度是否频繁。
- `page-faults`：模型加载、mmap、内存压力。

可加更具体事件：

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,page-faults,context-switches -- \
  ./build-cpu-prof/bin/llama-bench -m "$MODEL" -ngl 0 -p 128 -n 16 -t 14 -r 1
```

### perf record/report：看热点函数

```bash
perf record \
  -F 199 \
  --call-graph fp \
  -o profiles/perf/llama_cpu.data \
  -- ./build-cpu-prof/bin/llama-bench \
    -m "$MODEL" \
    -ngl 0 \
    -p 128 \
    -n 16 \
    -t 14 \
    -r 1

perf report \
  -i profiles/perf/llama_cpu.data \
  --stdio \
  --sort comm,dso,symbol \
  | tee profiles/perf/llama_cpu_report.txt
```

如果调用栈不完整：

- 确认使用 `RelWithDebInfo`。
- 确认加了 `-fno-omit-frame-pointer`。
- 尝试 `--call-graph dwarf`，但开销更大：

```bash
perf record -F 99 --call-graph dwarf,16384 -o profiles/perf/llama_cpu_dwarf.data -- \
  ./build-cpu-prof/bin/llama-bench -m "$MODEL" -ngl 0 -p 128 -n 16 -t 14 -r 1
```

### 观察线程数

线程太多可能让 token generation 变慢。可以尝试：

```bash
./build-cpu-prof/bin/llama-bench \
  -m "$MODEL" \
  -ngl 0 \
  -p 0 \
  -n 32 \
  -t 1,2,4,6,8,10,12,14,16,20 \
  -r 3 \
  -o jsonl \
  | tee profiles/bench/cpu_decode_threads.jsonl
```

如果 `-t 1` 比大线程数更快，说明 CPU 过饱和或内存带宽/调度是瓶颈。这一点在 GPU offload 时也可能发生：CPU 虽然不算大矩阵，但还负责调度、采样、同步。

### live server 的 perf

如果 server 已经在跑：

```bash
PID=$(pidof llama-server)
perf top -p "$PID"
```

抓 30 秒：

```bash
PID=$(pidof llama-server)
perf record -F 99 --call-graph fp -p "$PID" -o profiles/perf/server_30s.data -- sleep 30
perf report -i profiles/perf/server_30s.data --stdio | tee profiles/perf/server_30s_report.txt
```

## Profiling llama-server

`llama-server` 的性能不只是单请求 tokens/s，还包括并发、slot、连续 batching、KV cache、HTTP/JSON、流式输出。

### 构建并启动 server

如果 `build-cuda-prof/bin/llama-server` 不存在，先显式构建目标：

```bash
cmake --build build-cuda-prof --target llama-server -j
```

启动：

```bash
./build-cuda-prof/bin/llama-server \
  --host 127.0.0.1 \
  --port 8080 \
  -m "$MODEL" \
  -ngl 16 \
  -fa on \
  -c 4096 \
  -np 2 \
  -b 1024 \
  -ub 512 \
  --metrics \
  --slots \
  --perf \
  2>&1 | tee profiles/server/server.log
```

查询健康和指标：

```bash
curl -s http://127.0.0.1:8080/props | jq .
curl -s http://127.0.0.1:8080/slots | jq .
curl -s http://127.0.0.1:8080/metrics | tee profiles/server/metrics.txt
```

`/metrics` 里重点看：

- `llamacpp:prompt_tokens_total`
- `llamacpp:tokens_predicted_total`
- `llamacpp:prompt_tokens_seconds`
- `llamacpp:predicted_tokens_seconds`
- `llamacpp:kv_cache_usage_ratio`
- `llamacpp:requests_processing`
- `llamacpp:requests_deferred`
- `llamacpp:n_decode_total`
- `llamacpp:n_busy_slots_per_decode`

`n_busy_slots_per_decode` 接近 1，说明很多 decode 调用只有一个 slot 忙。并发压测时如果它长期低，连续 batching 可能没有发挥出来，或者请求分布不合适。

### 单请求压测

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "用三点解释 llama.cpp server profiling 应该看哪些指标。"}
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": false
  }' | jq .
```

### 并发压测

仓库自带 `tools/server/bench`，基于 k6。README 中说明需要带 SSE 扩展的 k6。简化学习时，也可以先用 `xargs -P` 做粗略并发：

```bash
seq 1 8 | xargs -I{} -P 4 curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "写一个 100 字以内的性能分析建议。"}],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": false
  }' > profiles/server/concurrent_responses.jsonl
```

压测期间另开终端观察：

```bash
watch -n 1 'curl -s http://127.0.0.1:8080/metrics | rg "prompt_tokens_seconds|predicted_tokens_seconds|requests|kv_cache|n_busy|n_decode"'
```

### 用 nsys 抓 server

server 是长进程，建议一边跑 nsys，一边另开终端发请求。先抓短时间：

```bash
nsys profile \
  --force-overwrite=true \
  --output=profiles/nsys/server_60s \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --backtrace=fp \
  --duration=60 \
  ./build-cuda-prof/bin/llama-server \
    --host 127.0.0.1 \
    --port 8080 \
    -m "$MODEL" \
    -ngl 16 \
    -fa on \
    -c 4096 \
    -np 2 \
    -b 1024 \
    -ub 512 \
    --metrics \
    --slots \
    --perf
```

另开终端在 60 秒内发请求。结束后看：

```bash
nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
  profiles/nsys/server_60s.nsys-rep \
  | tee profiles/nsys/server_60s_stats.txt
```

server timeline 重点看：

- HTTP worker 线程是否花很多时间在 JSON、模板、tokenization。
- `server_context` 主推理线程是否长期忙。
- `llama_decode` 调用之间是否有明显空洞。
- 并发请求是否聚成 batch，还是每个 slot 单独 decode。
- 流式输出是否造成频繁同步或小块发送开销。

## 源码阅读地图

当 profiler 指向一个热点时，可以按这张地图回到代码。

| 文件                                                  | 关注点                                                       |
| :---------------------------------------------------- | :----------------------------------------------------------- |
| `include/llama.h`                                     | `llama_decode()` API 语义、返回值、线程设置、KV/cache 参数   |
| `src/llama-context.cpp`                               | `llama_decode()` 包装、`llama_context::synchronize()`、内部 perf 统计 |
| `tools/llama-bench/llama-bench.cpp`                   | `test_prompt()`、`test_gen()`，benchmark 如何测 pp/tg        |
| `common/common.cpp`                                   | 初始化、warmup、`llama_perf_context_reset()`                 |
| `common/sampling.cpp`                                 | sampler 和输出计时                                           |
| `tools/server/server-context.cpp`                     | slot、batching、`update_slots()`、server metrics             |
| `tools/server/server-queue.cpp`                       | server task queue 和 update slots 调度                       |
| `ggml/src/ggml-backend.cpp`                           | backend scheduler、graph split、graph compute                |
| `ggml/src/ggml-cpu/ggml-cpu.c`                        | CPU graph compute、thread pool                               |
| `ggml/src/ggml-cuda/ggml-cuda.cu`                     | CUDA backend、CUDA Graphs、kernel launch、memcpy             |
| `ggml/src/ggml-cuda/fattn*.cu*`                       | Flash Attention CUDA kernels                                 |
| `ggml/src/ggml-cuda/mmq*.cu*`、`mmv*.cu*`、`mmf*.cu*` | CUDA matmul 路径                                             |

几个特别有用的位置：

- `src/llama-context.cpp` 中 `llama_context::synchronize()` 会把一次 decode 的耗时归到 prompt eval 或 eval。
- `src/llama-context.cpp` 中 `llama_perf_context_print()` 打印 load、prompt eval、eval 和 graphs reused。
- `tools/llama-bench/llama-bench.cpp` 中 `test_prompt()` 用随机 token 分 batch 调 `llama_decode()`；`test_gen()` 每个 token 调一次并同步。
- `tools/server/server-context.cpp` 中 `update_slots()` 会构造共享 batch，然后调用 `llama_decode()`，这是 server 的主要计算瓶颈。
- `examples/eval-callback/eval-callback.cpp` 展示了 `ggml_backend_sched_eval_callback`。它适合调试图和 tensor，不适合直接测性能，因为读取 tensor 会严重扰动执行。

## 常见瓶颈和实验矩阵

### 现象到原因

| 现象                    | 可能原因                                             | 下一步                               |
| :---------------------- | :--------------------------------------------------- | :----------------------------------- |
| prefill 慢，decode 正常 | batch 太小、BLAS/CUDA 未启用、Flash Attention 不合适 | 扫 `-b/-ub/-fa/-ngl`，看 nsys kernel |
| decode 慢，prefill 正常 | KV cache 长、CPU 同步、kernel launch overhead、采样  | 扫 `-d/-fa/-ctk/-ctv`，看 nsys 空洞  |
| GPU 利用率低            | CPU 瓶颈、partial offload、batch 太小、频繁同步      | nsys + perf                          |
| GPU memcpy 很多         | CPU/GPU 边界搬运、partial offload、logits 回读       | nsys memcpy report                   |
| `-t` 越大越慢           | CPU 过饱和、内存带宽竞争、P/E core 调度              | 线程扫描，必要时 CPU affinity        |
| 首 token 很慢           | 模型加载、prompt 太长、page faults、warmup           | 分离 load、prefill、warmup           |
| server 吞吐差           | slots 没聚批、请求太短、KV cache 紧张、HTTP 层重     | `/metrics`、`/slots`、nsys server    |
| 结果波动大              | 温度/功耗、后台任务、page cache、profiler 开销       | 固定环境，多次重复                   |
| OOM                     | offload 层太多、ctx 太大、KV 类型太大、f16 模型太大  | 降 `-ngl/-c`，量化模型，KV 量化      |

### 实验矩阵

| 假设                             | 改什么        | 命令方向                      | 期望信号                       |
| :------------------------------- | :------------ | :---------------------------- | :----------------------------- |
| CPU 线程过多拖慢 decode          | `-t`          | `-t 1,2,4,8,14,20`            | 找到吞吐峰值                   |
| prompt batch 不够                | `-b/-ub`      | `-n 0 -p 1024 -b ... -ub ...` | prefill t/s 随 batch 上升      |
| GPU offload 不足                 | `-ngl`        | `-ngl 0,8,16,24`              | t/s 提升，memcpy 变化          |
| Flash Attention 对长上下文有利   | `-fa` 和 `-d` | `-fa 0,1 -d 2048,4096`        | 长上下文 decode 改善           |
| KV cache 精度影响显存            | `-ctk/-ctv`   | `f16,q8_0,q4_0`               | VRAM 降低，速度变化            |
| CUDA Graphs 降低 launch overhead | build option  | graphs on/off                 | nsys 中 API launch 减少        |
| server batching 不充分           | `-np`、并发数 | k6 或并发 curl                | `n_busy_slots_per_decode` 上升 |
| 加载受 page cache 影响           | 冷热启动      | 连跑两次 CLI                  | 第二次 load time 明显下降      |

## 最后的工作流

推荐固定成一个循环：

1. 用 `llama-bench` 建立 baseline
2. 用内部 `--perf` 判断 load、prefill、decode 哪段慢。
3. 用 Nsight Systems 看 CPU/GPU 时间线和同步空洞。
4. 用 Nsight Compute 只分析 nsys 里最热的少数 CUDA kernel
5. 用 `perf` 分析 CPU 热点、采样、server HTTP/JSON、线程调度
6. 改一个变量，重新跑 baseline
7. 把命令、环境、指标、结论写进实验记录

真正有价值的 profiling 不是跑出一张漂亮图，而是把每个优化动作和可复现实验连接起来：为什么改、改了什么、指标如何变化、还有什么副作用。
