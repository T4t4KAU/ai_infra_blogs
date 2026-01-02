# How to Evaluate a vLLM Service Using EvalScope

书接上回：[How to Deploy the Qwen LLM on Muxi GPUs](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/misc/doc/deploy_qwen_on_muxi.md)

我们看看如何用EvalScope测试所部署模型的性能。

首先启动vLLM服务：

```powershell
vllm serve /models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --disable-log-requests
```

注意，这个命令和之前有所改动，主要是考虑到：

| 改动                     | 原因                        |
| ------------------------ | --------------------------- |
| `--served-model-name`    | EvalScope 的 `--model` 要用 |
| `--max-model-len 16384`  | 覆盖 2k / 4k / 16k 测试     |
| `--disable-log-requests` | 压测时减少 CPU 干扰         |

安装evalscope：

```
pip install -U evalscope
```

执行：

```powershell
evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model qwen3-32b \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --tokenizer-path /models/Qwen3-32B \
  --extra-args '{"ignore_eos": true}'
```

关键参数解释：

`--parallel 1 10 50 100 200`

EvalScope 会依次跑 5 轮测试：

| 测试轮次 | 并发数 |
| :------: | :----: |
| 第 1 轮  |   1    |
| 第 2 轮  |   10   |
| 第 3 轮  |   50   |
| 第 4 轮  |  100   |
| 第 5 轮  |  200   |

`--number 10 20 100 200 400`

表示每轮测试的总请求数

与 `--parallel` 一一对应：

| 并发 | 总请求 |
| :--: | :----: |
|  1   |   10   |
|  10  |   20   |
|  50  |  100   |
| 100  |  200   |
| 200  |  400   |

`--min-prompt-length 1024`

`--max-prompt-length 1024`

输入 prompt 长度固定为 1024 tokens，EvalScope 会随机生成 **1024 token 的 prompt**，做『固定负载』测试。这样做的好处是消除 prompt 长度波动，更容易对比不同并发下的性能。

`--max-tokens 1024`

`--min-tokens 1024`

生成长度固定为 1024 tokens，每个请求都会生成 **1024 token**

`--prefix-length 0`

表示Prompt 前缀长度，0 表示不使用共享前缀，每个请求完全独立。如果设成非 0，EvalScope 会复用 prefix，KV cache 利用率更高，更接近真实多轮对话。

`--dataset random`

使用 **随机 token 数据集**，不依赖真实语料，可专门用于性能测试，避免 tokenizer / I/O 干扰。

`--tokenizer-path /models/Qwen3-32B`

指定 tokenizer 路径，EvalScope 会用这个 tokenizer 计算 token 数，保证 **prompt_len / gen_len 准确**，tokenizer 必须和 vLLM 使用的一致，否则长度统计会错。

静候一段时间，可以看到最后一条日志，结果被保存到：

```
INFO: Performance summary saved to: outputs/20260102_110437/qwen3-32b/performance_summary.txt
```

可以看到：

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ Performance Test Summary Report                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Basic Information:
┌───────────────────────┬──────────────────────────────────────────────────────┐
│ Model                 │ qwen3-32b                                            │
│ Test Dataset          │ random                                               │
│ Total Generated       │ 197,632.0 tokens                                     │
│ Total Test Time       │ 507.05 seconds                                       │
│ Avg Output Rate       │ 389.77 tokens/sec                                    │
│ Output Path           │ outputs/20260102_110437/qwen3-32b                    │
└───────────────────────┴──────────────────────────────────────────────────────┘


                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│    1 │ 0.03 │   29.793 │   29.964 │   34.37 │    0.218 │   0.263 │    0.029 │   0.029 │    100.0%│
│   10 │ 0.19 │   32.257 │   32.290 │  190.25 │    0.802 │   1.105 │    0.031 │   0.031 │     30.0%│
│   50 │ 0.68 │   43.902 │   44.128 │  695.76 │    2.924 │   5.178 │    0.040 │   0.042 │     30.0%│
│  100 │ 0.85 │   53.399 │   53.776 │  875.21 │    4.329 │   7.973 │    0.048 │   0.051 │     23.0%│
│  200 │ 1.28 │   77.558 │   78.519 │ 1312.22 │    9.231 │  17.634 │    0.067 │   0.074 │     25.2%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 200 (1.28 req/sec)         
 Lowest Latency      Concurrency 1 (29.793 seconds)         

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
• Success rate is low at high concurrency, check system resources or reduce concurrency
```

关键信息在于这个表格。

首先是**RPS（Request per Second）**

| 并发 | RPS  |
| :--: | :--: |
|  1   | 0.03 |
|  10  | 0.19 |
|  50  | 0.68 |
| 100  | 0.85 |
| 200  | 1.28 |

可见RPS **随着并发上升而上升**，但不是线性上升的。

**Avg Latency / P99 Latency**

| 并发 | Avg Lat(s) | P99 Lat(s) |
| :--: | :--------: | :--------: |
|  1   |    29.8    |   29.964   |
|  10  |    32.3    |   32.290   |
|  50  |    43.9    |   44.128   |
| 100  |    53.4    |   53.776   |
| 200  |    77.6    |   78.519   |

可见P99 ≈ Avg，**所有成功请求都跑满了完整生成**，后进请求只能等前面的 decode 结束。

Avg Latency 是 所有『成功请求』的延迟平均值；P99 Latency 的含义是99%的时间都比这个值快，所以这个指标揭示了最差体验。

**Gen toks/s（生成吞吐）**

| 并发 | tok/s |
| :--: | :---: |
|  1   |  34   |
|  10  |  190  |
|  50  |  696  |
| 100  |  875  |
| 200  | 1312  |

多请求的吞吐量相对于单请求的吞吐量上涨很多，说明vLLM 的 **continuous batching 是生效的**，GPU 的 **真实工作量在并发升高时被吃满**。

**TTFT（Time To First Token）**

| 并发 | Avg TTFT |
| :--: | :------: |
|  1   |  0.22s   |
|  10  |  0.80s   |
|  50  |  2.92s   |
| 100  |  4.33s   |
| 200  |  9.23s   |

这个指标对用户体验来说很重要，TTFT ≈ **排队时间 + prefill**，随并发线性恶化。

**TPOT（Time Per Output Token）**

| 并发 | TPOT   |
| ---- | ------ |
| 1    | 0.029s |
| 200  | 0.067s |

反映生成一个 token平均要花多少时间，整体变化不大

**Success Rate**

| 并发 | 成功率 |
| :--: | :----: |
|  1   |  100%  |
|  10  |  30%   |
|  50  |  30%   |
| 100  |  23%   |
| 200  |  25%   |

并发 ≥10 后，系统已经进入『不可用状态』。

可以总结：

|    目标     |    合理并发    |
| :---------: | :------------: |
|  最低延迟   |       1        |
|  可用服务   |      2–8       |
| 压 GPU 吞吐 | 50–100（离线） |
|  线上 API   |    不能 ≥10    |

这句话揭示了这个系统的关键问题：

```
Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
• Success rate is low at high concurrency, check system resources or reduce concurrency
```

GPU 计算吞吐还没到极限，但『服务系统』已经先扛不住了，

EvalScope 是从**吞吐视角**得出的判断：`Gen tok/s` 和`Total tok/s` 随着并发上升，还没看到平台期。要知道真实的瓶颈还得进一步分析。

