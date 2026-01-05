# Optimizing a CUDA Softmax Kernel: A Deep Dive into Reductions, Memory, and Warp-Level Primitives

本文讲解如何一步一步用CUDA实现一个高性能的Softmax（单精度矩阵乘法）算子。

## Softmax 算法

Softmax 把一组任意实数，变成『可比较的概率分布』。

做了三件事：

1. 每个值都变成 **非负**
2. 所有值 **和为 1**
3. 大的值会被 **指数级放大**，小的值被压缩

给定一个向量：
```math
\mathbf{z} = (z_1, z_2, \dots, z_n)
```
Softmax 定义为：
```math
\text{Softmax}(z_i)
= \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
```
输出：
```math
\mathbf{p} = (p_1, p_2, \dots, p_n)
```

为什么用指数函数？因为：

- 保证正数

$$
e^{z_i} > 0
$$

- 放大差异，softmax 会让『最大 logit』主导分布：

  - 差 1 → 比例 ≈ 2.7 倍

  - 差 5 → 比例 ≈ 148 倍

- 可微，意味着可以反向传播，适合用在神经网络中

 Softmax 的核心性质：

- 输出是概率分布：

$$
0 < p_i < 1,\quad \sum_i p_i = 1
$$

- 平移不变性（数值稳定关键）：

$$
\text{Softmax}(z)=\text{Softmax}(z - c)
$$

- 不具备尺度不变性：
  $$
  \text{Softmax}(\alpha z) \neq \text{Softmax}(z)
  $$

要注意的是，Softmax中的指数函数增长是非常快的：

- $e^{10} \approx 2.2 \times 10^4$
- $e^{50} \approx 5.2 \times 10^{21}$
- $e^{100} \approx 2.7 \times 10^{43}$

但 **float32** 能表示的最大值大约是：

$$
\text{max float32} \approx 3.4 \times 10^{38}
$$

也就说，一旦 $z_i \gtrsim 88$ ， $e^{z_i}$ **直接溢出成 `Inf`**，这显然是不可接受的，于是利用平移不变性，我们像如下改造公式：

$$
\text{Softmax}(z_i)
= \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

显然有：

$$
e^{z_i - \max(z)} \le 1
$$

因此，不可能再溢出。

Softmax在LLM中有非常重要的用途，主要体现在Attention 权重归一化。

在自注意力计算中，有如下公式：

$$
A = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

把每个 Query 对所有 Key 的相似度，转换成 **『注意力权重分布』**，每一行表示"我该关注谁"，强调最相关 token。

## V1 最朴素Softmax计算

可以很轻松地实现一个最朴素版本的Softmax Kernel：

```c
// CUDA kernel for computing the softmax function.
// Each thread processes one row of the input matrix.
__global__ void softmax_forward_kernel_v1(float *__restrict__ output,      // [N, C] output tensor
                                          const float *__restrict__ input, // [N, C] input tensor
                                          int num_rows,                    // N: number of rows
                                          int num_cols                     // C: number of columns per row
) {
    // Global thread index corresponding to the row index
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds threads
    if (row_idx >= num_rows) {
        return;
    }

    // Pointers to the current row
    const float *input_row = input + row_idx * num_cols;
    float *output_row = output + row_idx * num_cols;

    // Step 1: find the maximum value in the row (for numerical stability)
    float max_value = -CUDART_INF_F;
    for (int col = 0; col < num_cols; ++col) { max_value = fmaxf(max_value, input_row[col]); }

    // Step 2: compute exponentials and their sum
    float sum_exp = 0.0f;
    for (int col = 0; col < num_cols; ++col) {
        float exp_val = expf(input_row[col] - max_value);
        output_row[col] = exp_val;
        sum_exp += exp_val;
    }

    // Step 3: normalize to obtain probabilities
    float inv_sum = 1.0f / sum_exp;
    for (int col = 0; col < num_cols; ++col) { output_row[col] *= inv_sum; }
}
```

我们让每个线程处理一整行，遍历行中所有元素，找到行中的最大值，接着求和算出分母，代入计算后得到最终结果。

用Nsight Compute分析一下这个kernel的性能：

![softmax_v1_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_1.png)

可以看到，Nsight表明："**This kernel grid is too small to fill the available resources on this device, resulting in only 0.06 full waves across all SMs.**"，GPU Speed Of Light Throughput中的指标数据也体现了这一点。这意味着我们的GPU中的SM很空闲，大量的算力都没用上，GPU 的理论峰值完全用不上。

这里说"Small Grid"，本质上是因为并行映射方式的问题。测试中，我们设定N=C=4096，grid.x 计算方式是：

```c
grid.x = ceil(N / blockDim.x)
```

同时设置：

```c
N = 4096
blockDim.x = 128
```

于是得到：grid.x = 4096 / 128 = 32 blocks

Nsight判断是否是Small Grid，是看：能不能在所有 SM 上形成足够多的 "waves"，wave指的是：

- **1 wave** ≈ *所有 SM 同时至少跑 1 个 block*
- **full wave** ≈ GPU 的每个 SM 都在干活

0.06 full waves 就表明了大量SM空闲。

再从代码角度分析，这个kernel为什么这么慢。

首先是这段代码：

```c
// Step 1: find the maximum value in the row (for numerical stability)
float max_value = -CUDART_INF_F;
for (int col = 0; col < num_cols; ++col) { max_value = fmaxf(max_value, input_row[col]); }
```

这段代码是顺序遍历一整行，找到最大值，实际上这一步骤可以并行执行。

求和也是一样：

```c
// Step 2: compute exponentials and their sum
float sum_exp = 0.0f;
for (int col = 0; col < num_cols; ++col) {
    float exp_val = expf(input_row[col] - max_value);
    output_row[col] = exp_val;
    sum_exp += exp_val;
}
```

同样是遍历，这部分也可以并行化。

## V2 Shared Memory & Block Reduce

为了优化上述两个部分，我们基于共享内存实现块内加速。

可以这样计算计算最大值，每个线程先计算局部最大值：

```c
float local_max = -CUDART_INF_F;
for (int col = tid; col < num_cols; col += block_size) {
    local_max = fmaxf(local_max, input_row[col]);
}
```

每个线程实际上处理：

```
thread 0  → columns: 0, block_size, 2*block_size, ...
thread 1  → columns: 1, block_size+1, 2*block_size+1, ...
...
```

每个线程隔一个 block_size 取一个元素，一次性处理多个元素，叫作线程粗化（thread coarsening）。local_max是每个线程计算出的局部最大值。

把局部最大值写入 shared memory：

```c
shared[tid] = local_max;
```

计算完局部最大值后，就开始并行规约：

```c
for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
}
```

每一轮规约，参与的线程数量减半，这是一个**树形并行规约（tree reduction）**：

- 第 1 轮：
  - 128 → 64 个值
- 第 2 轮：
  - 64 → 32
- …
- 最后：
  - 1 个值（shared[0]）

最终得到：

```c
shared[0] = max(input_row[0..num_cols-1])
```

求和的做法是相近的，如下所示：

```c
// Broadcast the maximum value to all threads
float max_value = shared[0];

// ------------------------------------------------------------------
// Step 2: compute exponentials and their sum
// Each thread again processes multiple columns
// ------------------------------------------------------------------
float local_sum = 0.0f;
for (int col = tid; col < num_cols; col += block_size) {
    float exp_val = expf(input_row[col] - max_value);
    output_row[col] = exp_val;
    local_sum += exp_val;
}

// Write partial sums to shared memory
shared[tid] = local_sum;
__syncthreads();

// Block-level reduction to compute the sum of exponentials
for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    __syncthreads();
}

// Broadcast the sum to all threads
float sum_exp = shared[0];
```

计算流程大致如下：

![softmax_v2_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_3.png)

使用Nsight分析该kernel：

![softmax_v2_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_4.png)

可以看到性能提升了相当多，Small Grid的问题也消失了，并行度大大提高，主要问题来到了 **内存子系统（尤其是 DRAM）**，那么下面就着手针对这一点进行优化。

注意到有一栏：

![softmax_v2_3](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_6.png)

这个数据揭示了"到底是哪几条指令，在 warp stall 中贡献最大？"，点击第一条被认为是最多贡献的指令，我们来到新的页面。

通过这些更底层的数据，可以从**GPU 执行层（SASS）+ 微架构调度层**看到更为细致的数据：

![softmax_v2_4](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_5.png)

这张图的最左侧一栏展现的是SASS 指令：

```
IMAD.WIDE.U32
LDG.E.CONSTANT
FFMA / FMNMX
BRA
```

可以简单了解一些指令：

| 指令               | 含义                                        |
| ------------------ | ------------------------------------------- |
| `IMAD.WIDE.U32`    | 32-bit integer multiply-add（地址计算常见） |
| `LDG.E.CONSTANT`   | 从 constant / global memory 读              |
| `FMNMX`            | 浮点 min/max（softmax 的 max reduction）    |
| `FFMA / FADD`      | 浮点运算                                    |
| `BRA`              | 分支                                        |
| `BSYNC.RECONVERGE` | warp 分支重汇合                             |

中间蓝色条展现的是每个线程（thread）正在占用的寄存器数量，这条指令执行时，活跃的寄存器数量。

右侧的两列，**`Attributed Stalls`** 和 **`Warp Stall Sampling (Not-issued Samples)`** 表示：

|                     指标                     |        核心问题         |                           定义                            |
| :------------------------------------------: | :---------------------: | :-------------------------------------------------------: |
| **Warp Stall Sampling (Not-issued Samples)** | *warp 什么时候没发射？* | 统计warp 本应发射指令但没有发射的采样次数，也就是停顿次数 |
|            **Attributed Stalls**             |    *为什么没发射？*     |     尝试判断原因，把没发射的 stall 按原因归因后的结果     |

可以发现，有些指令的Attributed Warp Stall Sampling是空的，因为这些指令在采样时刻没有『Attributed Stall』，这些指令要么没有导致 warp stall， 要么 stall 已经被归因给了别的指令。考虑Nsight的测试机制，Nsight 的 stall sampling 只在 warp 没能发射指令时才进行采样，只给『导致 stall 的那条指令』归因，所以不是每条指令都会被采样，不是每条指令都会有 stall。

一条 SASS 指令**只有在同时满足下面条件时**，才可能显示 stall 百分比：

- warp 想执行下一条指令，但不能执行

- 原因能明确归因到某一类（FP / memory / barrier / scoreboard 等）

- 采样点正好落在这个等待阶段

纯算术、无依赖的指令一般没有Attributed Stall，因为延迟非常低，warp 执行完后，下一条指令立刻 ready。

更多详细信息可参考：https://docs.nvidia.com/nsight-compute/ProfilingGuide/

我们点击对Warp Stall贡献最多的指令，可以看到：

![softmax_v2_5](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_7.png)

Scoreboard Stalls = **44.6% (7.18K)**，表示在所有由于scoreboard（寄存器依赖）而产生的 stall 采样中，有 44.6% 的 stall 是由 Floating Point 指令引起的，对应约 7,180 次采样，也就说：**warp 卡住的时候，将近一半是在等浮点指令的结果写回寄存器。**

Nsight在这里会指示scoreboard 等待的最后一个写寄存器的指令是什么类型，所以指向了第70行，作为『输入依赖源』：

```
LDG.E.CONSTANT R12, desc[UR8][R12.64]					
```

第 70 行被标记，是因为它是 scoreboard 等待寄存器的最近写入者，但是这并不表明load是性能瓶颈的"罪魁祸首"，因为这条指令的Attributed Warp Stall Sampling很低，主要原因还是FP依赖，这从代码逻辑中就可以推断出来。

结合代码，这些指令实际上发生在：

```c
// Write partial maxima to shared memory
shared[tid] = local_max;
__syncthreads();

// Block-level reduction to find the maximum value
for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }

    __syncthreads();
}
```

其特性是：

- 下一次迭代 **必须等待** 上一次结果
- warp 内无法并行
- scheduler 无法打散

这里的循环携带依赖才是FMNMX被停顿的根本原因，导致了大量的FP依赖。虽然定位到了问题，但是很遗憾，笔者没有能力优化这个瓶颈，这是一个典型的二叉树规约，这一层的数据要依赖上一层的数据这顺理成章，笔者并无先验知识可以消除这个依赖🥹

## V3 Warp Shuffle Instructions

不过好在，我们可以对访存再做一步优化。

在这一版本代码中，我们引入一种特别的指令：洗牌指令，即shuffle指令。

洗牌指令一般形式如下：

```c
__shfl_xxx_sync(mask, value, src_lane, width)
```

参数如下：

|   属性   |          说明          |
| :------: | :--------------------: |
|   范围   |     仅限一个 warp      |
| 通信介质 |         寄存器         |
|   同步   |    warp 内隐式同步     |
|   延迟   |         非常低         |
| 适用场景 | 归约、扫描、广播、重排 |

我们知道，Warp 执行模型有：

- 一个 warp = 32 个线程
- warp 内 **SIMT 锁步执行**
- 每个线程有自己的寄存器

洗牌指令**打破了『线程只能访问自己寄存器』的限制**，但只在 warp 内有效。

 `__shfl_sync`（任意 lane 访问）

```c
int v = __shfl_sync(0xffffffff, x, srcLane);
```

所有线程从 `srcLane` 线程读取 `x`，用于广播 / 收集数据非常高效

`__shfl_up_sync`（向上移动）

```c
int v = __shfl_up_sync(mask, x, delta);
```

从 `laneId - delta` 读取，如果laneId < delta，函数会返回未定义

`__shfl_down_sync`（向下移动）

```c
int v = __shfl_down_sync(mask, x, delta);
```

从 `laneId + delta` 读取，常用于归约（reduce）

 `__shfl_xor_sync`（蝶形交换）

```c
int v = __shfl_xor_sync(mask, x, laneMask);
```

`srcLane = laneId ^ laneMask`，非常适合 **树形 / butterfly 结构**

这类指令非常用于块内规约，一个block 内先用 shuffle，再用 shared memory 合并 warp：

```c
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

以求和为例，**warp 内的每个线程**从 **lane_id + offset** 的线程那里，读取它的 `val` 寄存器值，遇到越界的读取直接返回原值。

| 当前线程 lane_id |    读取谁的 val    |
| :--------------: | :----------------: |
|        0         |       lane 1       |
|        1         |       lane 2       |
|        …         |         …          |
|        30        |      lane 31       |
|        31        | （越界，值未定义） |

这一操作中：

- 没有任何内存访问
- 没有 shared memory
- 没有同步

由此实现了一个**标准的 warp 级二叉树规约（tree reduction）**

 第 1 轮：offset = 16

```c
val += value from lane_id + 16
```

结果：

- lane 0 得到：`v0 + v16`
- lane 1 得到：`v1 + v17`
- …
- lane 15 得到：`v15 + v31`
- lane 16–31：结果无意义（后面不会再用）

从32个元素中得到16 个有效 partial sum，

第 2 轮：offset = 8

```c
val += value from lane_id + 8
```

- lane 0：`(v0+v16) + (v8+v24)`
- lane 1：`(v1+v17) + (v9+v25)`
- …

从16个元素得到8个和，接着以此类推。

将新的Reduce替换到原来的代码中：

```c
// ------------------------------------------------------------------
// Shared memory layout:
// [0 ... warps_per_block - 1]          -> warp max values
// [warps_per_block ... 2*warps_per_block - 1] -> warp sum values
// ------------------------------------------------------------------
extern __shared__ float shared[];
float *warp_max = shared;
float *warp_sum = shared + warps_per_block;

// ------------------------------------------------------------------
// Step 1: compute maximum value of the row (numerical stability)
// Thread coarsening + warp-level reduction
// ------------------------------------------------------------------
float local_max = -CUDART_INF_F;
for (int col = tid; col < num_cols; col += block_size) {
    local_max = fmaxf(local_max, input_row[col]);
}

// Warp-level max reduction
local_max = warpReduceMax(local_max);

// Write warp result to shared memory
if (lane_id == 0) {
    warp_max[warp_id] = local_max;
}
__syncthreads();

// Block-level reduction across warps
if (tid == 0) {
    float max_val = warp_max[0];
    for (int i = 1; i < warps_per_block; ++i) {
        max_val = fmaxf(max_val, warp_max[i]);
    }
    warp_max[0] = max_val;
}
__syncthreads();

float max_value = warp_max[0];

// ------------------------------------------------------------------
// Step 2: compute exponentials and their sum
// Thread coarsening + warp-level reduction
// ------------------------------------------------------------------
float local_sum = 0.0f;
for (int col = tid; col < num_cols; col += block_size) {
    float exp_val = expf(input_row[col] - max_value);
    output_row[col] = exp_val;
    local_sum += exp_val;
}

// Warp-level sum reduction
local_sum = warpReduceSum(local_sum);

// Write warp sum to shared memory
if (lane_id == 0) {
    warp_sum[warp_id] = local_sum;
}
__syncthreads();

// Block-level reduction across warps
if (tid == 0) {
    float sum_val = warp_sum[0];
    for (int i = 1; i < warps_per_block; ++i) {
        sum_val += warp_sum[i];
    }
    warp_sum[0] = sum_val;
}
__syncthreads();

float sum_exp = warp_sum[0];
```

取缔了原来对共享内存的频繁使用。

但是出人意料的是，在初步的测试中，V3和V2的执行时间居然几乎一样！🤨

但是我们如果调大V2和V3的`block_size`，从128设置到1024，却可以发现V3的性能相较V2大大提高。

![softmax_v3_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_8.png)

![softmax_v3_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/softmax_9.png)

那是因为block_size很大时，则Reduce的轮数大大增高，V2的同步开销才暴露出来，这样一来才体现出V3的优势。

