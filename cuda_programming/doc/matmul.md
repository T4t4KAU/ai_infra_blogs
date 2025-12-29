# Building a High-Performance CUDA SGEMM Kernel: From Naive Implementation to Warp Tiling

本文讲解如何一步一步用CUDA实现一个高性能的SGEMM（单精度矩阵乘法）算子。

实验环境:

| 配置 |             型号             |
| :--: | :--------------------------: |
| CPU  | Intel(R) Core(TM) i5-14600KF |
|  OS  |         Ubuntu 24.04         |
| GPU  |       NVIDIA RTX 5070        |
| CUDA |             13.0             |

## V1 最朴素版矩阵运算

源代码：[matmul_v1.cu](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/src/matmul_v1.cu)

已知矩阵运算公式：
```math
C_{m,n}=\sum_{k=0}^{K-1}A_{m,k}\cdot B_{k,n}
```
从数学上看，矩阵运算是天然可以并行的，这样的运算可以分解，每个输出元素都是一个独立任务，求和也同样可以并行，并且这些并行的计算都较为简单，无控制逻辑。于是，可以轻易地想到，我们可以利用GPU的大规模并行计算能力来高校地完成矩阵运算。

在单精度矩阵乘法中，公式一般写成：
```math
C=\alpha\cdot A\cdot B+\beta\cdot C
```
于是，可以先写出如下代码：

```c++
__global__ void sgemm_kernel_v1(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum_val = 0.;

        for (int k = 0; k < K; ++k) {
            sum_val += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = alpha * sum_val + beta * C[row * N + col];
    }
}
```

我们想让一个cuda线程负责计算C中的一个元素，这样要计算出M行N列的矩阵，要用到M×N个线程。

`__global__`定义了一个GPU kernel函数，表示这个函数由CPU启动，在GPU上由数个线程同时执行。

考虑到矩阵是2D数据结构，因此用2D线程块映射矩阵数据。`blockIdx.x`表示线程块在整个GPU网格中的x方向上的索引，`blockDim.x`表示线程块在x方向的大小，`threadIdx.x`表示在线程块中的x方向上属于第几个线程。

CUDA的线程分层模型可以参见：[NVIDIA DOC | Thread Hierarchy](https://docs.nvidia.cn/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)

由此可知，下面两行其实是算执行这个kernel的线程在全局网格中的(x,y)坐标，以此确定这个线程要计算矩阵中哪一个元素：

```c++
int col = blockIdx.x * blockDim.x + threadIdx.x; // 要计算元素在矩阵中的列
int row = blockIdx.y * blockDim.y + threadIdx.y; // 要计算元素在举证中的行
```

要注意的是，GPU上的线程块是整块整块启动的，不一定能刚好覆盖矩阵，例如一个线程块是16×16，一个矩阵可以是16×18，那么至少得用两个线程块完成计算，于是会有一部分线程落在矩阵边界外，因此要注意越界访问：

```c++
if (row < M && col < N) { // 检查是否越界
	....
}
```

接下来按照矩阵运算公式实现计算即可。

不难发现，这样的实现在性能上存在一些问题：

1. 频繁读取全局内存，所有的A和B元素都是从全局内存中读取，这样浪费了太多时间在访存上，计算效率低下
2. 计算强度低，也就是算的少，读的多，每从显存搬 8 个字节，只做了 2 次计算，计算单元被大量空闲
3. 缺乏数据复用，每个线程都从内存中读取A的一整行，B的一整列，这类数据重复读取，属于冗余操作

可以用NVIDIA提供的[Nsight Compute](https://developer.nvidia.com/nsight-compute)来分析，在当前目录下执行命令，会在终端输出大量数据：

```bash
nvcc -O3 matmul_v1.cu -o matmul # 编译代码
ncu ./matmul # 要用root权限，注意当前的ncu版本能否用于当前显卡
```

也可以使用将数据可视化，先执行：

```bash
ncu -o sgemm_v1 ./matmul #在目录下会生成一些文件
ncu-ui # 会打开一个窗口
```

在弹出的窗口中选择"File"->"Open File"然后选择生成的文件sgemm_v1.ncu-rep，之后就可以看到分析结果。

关注『Section: GPU Speed Of Light Throughput』，发现L1 Cache的负载高达90%以上。可知L1 Cache几乎被打满，说明对全局内存的访问压力确实很大，接下来就着手优化这一点。

如下是测试结果：

![matmul_v1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_1.png)

要补充的是，在做性能测试时应该测试的是kernel稳定运行时的性能，所以应该排除第一次的冷启动，详情见实际代码的注释。

## V2 Thread Tile优化

源代码：[matmul_v2.cu](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/src/matmul_v2.cu)

在这一版本代码中，我们要对原始的代码进行优化，观察到访问全局内存过于频繁，很容易想到用共享内存来避免频繁访问全局内存，同时也能实现数据复用。

同时，在计算上我们要用分块矩阵乘法，让一个线程块去计算矩阵C的一个分块（BM×BN矩阵），可作如下推导：

有矩阵C：
```math
C=
\begin{bmatrix}
C_{0,0} & C_{0,1} & \cdots \\
C_{1,0} & C_{1,1} & \cdots \\
\vdots
\end{bmatrix}
```
设矩阵C的一个分块Block的坐标是(p,q)，则有：
```math
C^{(p,q)}\triangleq
\begin{array}
{c}C[pBM:(p+1)BM-1,\space
\end{array}qBN:(q+1)BN-1]
```


与之对应的 A Block 与 B Block：
```math
A^{(p)}\triangleq A[pBM:(p+1)BM-1,\mathrm{~}0:K-1] \\
B^{(q)}\triangleq B[0:K-1,qBN:(q+1)BN-1]
```
于是C Block可以这样得到：
```math
C^{(p,q)}=A^{(p)}\cdot B^{(q)}
```
如下图所示：

![matmul_v2_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_3.png)

上面的大分块称之为**Block**，接下来将K按BK长度也做分割，得到的小分块称为**Tile**，那么可以如下算出完整的C Block：
```math
C_{block}=\sum_{t=0}^{K/BK-1}\left(A_{block}^{(t)}\cdot B_{block}^{(t)}\right)
```
其中，t标识了这个Tile在Block中的索引：
```math
A_{block}^{(t)}\in\mathbb{R}^{BM\times BK} \\
B_{block}^{(t)}\in\mathbb{R}^{BK\times BN}
```
与之对应的A Tile和B Tile：
```math
A^{(p,t)}\triangleq A[pBM:(p+1)BM-1,\mathrm{~}tBK:(t+1)BK-1] \\
B^{(q,t)}\triangleq B[tBK:(t+1)BK-1,\mathrm{~}qBN:(q+1)BN-1]
```
于是，C Block的第t个Tile中每个元素的计算如下：
```math
\boxed{C^{(p,q,t)}\triangleq A^{(p,t)}\cdot B^{(q,t)}}\quad\Rightarrow\quad C^{(p,q,t)}\in\mathbb{R}^{BM\times BN}
```
将所有C Tile求和，得到最后的完整C Block：
```math
C^{(p,q)}=\sum_{t=0}^{T-1}C^{(p,q,t)}=\sum_{t=0}^{T-1}A^{(p,t)}\cdot B^{(q,t)}
```
可通过下图理解：

![matmul_v2_2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_4.png)

上面提到，要用一个线程块计算一个Block，那么如何将一个Block分配给线程块中的多个线程？

先定义线程的逻辑坐标是(u, v)，并让一个线程计算出一个TM×TN分块，则坐标满足：
```math
\begin{aligned}
u & =0,1,\ldots,\frac{BM}{TM}-1 \\
v & =0,1,\ldots,\frac{BN}{TN}-1
\end{aligned}
```
于是一个线程负责算出如下小分块：
```math
C_{u,v}^{(p,q)}\triangleq C^{(p,q)}[uTM:(u+1)TM-1,\mathrm{~}vTN:(v+1)TN-1]
```
也就是：

![matmul_v2_3](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_5.png)

如上所示，我们将一个矩阵运算划分成了三层结构，先对M行N列划分，将C划分出BM×BN的C Block，将A划分成BM×K的A Block，将B划分成K×BN的B Block；接着对K进行划分，A Block划分成BM×BK的A Tile，B Block划分出BM×BK的B Tile；最后考虑线程分配，将Block划分出TM×TN的块，由一个线程负责运算。

**这样划分有什么好处？**

第一步划分是为了让一个线程块计算一个C Block，一个线程块内的线程可以共享数据，这样就实现了数据复用。可以将要用的矩阵块先读到共享内存中，在线程块中共享，这样A的一行可以被BN个列重复使用，B的一列可以被BM个行重复使用，就不像V1版本的代码，每个线程做一次运算都要读取一整行，一整列，完全没用到线程块的数据复用特性。不过这个BM和BN的大小要考虑到共享内存的容量进行设置。

第二步划分是考虑到共享内存容量有限，K可能会很大，共享内存中根本放不下，所以得将K分割。

第三步划分是考虑到指令并行，决定一个线程算哪些元素，这样也能利用好线程自身的寄存器。

接下来演示代码如何编写。

首先是确定当前线程块应该计算C Block的哪一部分：

```c
/* ------------------------------
* Block coordinates
* Each block computes one BM×BN tile of C
* ------------------------------ */     
const int block_col = blockIdx.x; // index in N dimension
const int block_row = blockIdx.y; // index in M dimension
```

这个由Thread Block的坐标决定即可。

接下来确定要用到的线程数：

```c
/* ------------------------------
* Thread layout inside a block
* Threads are logically arranged in 2D:
*   (BM / TM) × (BN / TN)
* ------------------------------ */
const int threads_per_row = BN / TN;
const int threads_per_col = BM / TM;
const int num_threads = threads_per_row * threads_per_col;
```

确定当前线程要算C Block中的哪一部分（计算具体位置）：

```c
/* ------------------------------
* Per-thread output tile offset
* Each thread computes a TM×TN sub-tile of C
* ------------------------------ */
const int tx = (threadIdx.x % threads_per_row) * TN; // column offset
const int ty = (threadIdx.x / threads_per_row) * TM; // row offset
```

创建共享内存：

```c
/* ------------------------------
* Shared memory for A and B tiles
* As : BM × BK
* Bs : BK × BN
* ------------------------------ */
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];
```

拿到要用的A Block和B Block并将指针放到矩阵C的对应位置上：

```c
/* ------------------------------
* Move global pointers to the
* beginning of the current block tile
* ------------------------------ */    
float *A_block = A + block_row * BM * K;
float *B_block = B + block_col * BN;
float *C_block = C + block_row * BM * N + block_col * BN;
```

接下来要从全局内存加载要用的数据到共享内存：

```c
/* ------------------------------
* Thread mapping for loading A tile
* ------------------------------ */
const int a_tile_row = threadIdx.x / BK;
const int a_tile_col = threadIdx.x % BK;
const int a_tile_stride = num_threads / BK;

const int b_tile_row = threadIdx.x / BN;
const int b_tile_col = threadIdx.x % BN;
const int b_tile_stride = num_threads / BN;

#pragma unroll
for (int k = 0; k < K; k += BK) {
/* ------------------------------
 * Load A tile into shared memory
 * ------------------------------ */
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        As[(a_tile_row + i) * BK + a_tile_col] = A_block[(a_tile_row + i) * K + a_tile_col];
    }
/* ------------------------------
 * Load B tile into shared memory
 * ------------------------------ */
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        Bs[(b_tile_row + i) * BN + b_tile_col] = B_block[(b_tile_row + i) * N + b_tile_col];
    }

    __syncthreads(); // Ensure As and Bs are fully loaded


    /* ------------------------------
     * Advance A and B pointers
     * ------------------------------ */
    A_block += BK;
    B_block += BK * N;
}
```

每一轮加载一个A Tile和一个B Tile到共享内存，那么一共就有(K/BK)轮，每个线程负责搬运一部分数据，A Tile中的映射关系大致如下：

```
thread 0: A_block(0, 0) → As(0, 0), A_block(0 + stride, 0) → As(0 + stride, 0), ...
thread 1: A_block(0, 1) → As(0, 1), A_block(0 + stride, 1) → As(0 + stride, 1), ...
thread 2: A_block(0, 2) → As(0, 2), A_block(0 + stride, 2) → As(0 + stride, 2), ...
...
thread X: A_block(0, X) → As(0, X), A_block(0 + stride, X) → As(0 + stride, X)
...
thread BK: A_block(1, 0) → As(1, 0), A_block(1 + stride, 0) → As(1 + stride, 0), ...
thread (BK + 1): A_block(1, 1) → As(1, 1), A_block(1 + stride, 1) → As(1 + stride, 1), ...
...
```

最后要用`__syncthreads()`进行同步，确保所有线程都完成。加载完一个Tile后要把指针向前移动，便于下一轮加载后续的Tile到共享内存。

在每一轮加载Tile数据的同时，也要完成计算：

```c
#pragma unroll
    for (int k = 0; k < K; k += BK) {
    /* ------------------------------
     * Load A tile into shared memory
     * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A_block[(a_tile_row + i) * K + a_tile_col];
        }
    /* ------------------------------
     * Load B tile into shared memory
     * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B_block[(b_tile_row + i) * N + b_tile_col];
        }

        __syncthreads(); // Ensure As and Bs are fully loaded


        /* ------------------------------
            * Advance A and B pointers
            * ------------------------------ */
        A_block += BK;
        B_block += BK * N;

        /* ------------------------------
        * Compute: register-level GEMM
        * ------------------------------ */
#pragma unroll
        for (int t = 0; t < BK; ++t) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += As[(ty + i) * BK + t] * Bs[t * BN + (tx + j)];
                }
            }
        }
        __syncthreads();
    }
```

可以看到现在获取数据直接从共享内存中获得即可，一个线程计算出每个K-Tile中TM×TN的分块，再将所有K-Tile的结果加和，得到这一轮最终运算结果，即一个C Block中TM×TN分块的最终结果。

最后一步将TM×TN分块的结果代入并写回到C Block即可：

```c
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            // C_block(ty + i, tx + j) = alpha × accum(i, j) + beta × C_block(ty + i, tx + j)
            C_block[(ty + i) * N + (tx + j)] = alpha * accum[i][j] + beta * C_block[(ty + i) * N + (tx + j)];
        }
    }
```

在上述所有代码中，索引计算是一个难点，必须得谨慎处理。

下面看看Nsight Compute的分析结果：

![matmul_v2](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_2.png)

可以发现消耗的时间和缓存的负载都大大减少，计算密度和指令并行都得到了提高。

但是可以发现V1的测试结果中的算力理论峰值（Compute (SM) Throughput %）比V2高得多，不过这不代表V1算的更快或利用率更高，因为这个指标实际上是『计算指令发射率/理论最大发射率』，这个指标只反映"忙不忙"，而不反映"干了多少"。V1中每个线程只做了很简单的运算，故而指令发射比例非常高，但实际上每个线程干的很少；V2中的指令更复杂，用到了对共享内存的load/store以及同步指令，不是每个周期都在运行计算指令，所以指令发射率降低，因此这个指标实际上不体现计算速度。

## V3 向量化预取

源代码：[matmul_v3.cu](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/src/matmul_v3.cu)

一个warp的32个线程访问全局内存时，如果地址连续、对齐则GPU可以用最少的内存事务完成访问，即合并访存。

如下是32个线程访问连续的、对齐的地址，且使用的数据类型完全相同：

```
thread 0 → addr + 0
thread 1 → addr + 1
thread 2 → addr + 2
...
thread 31 → addr + 31
```

只需要1～2次内存事务就可以完成该访问，带宽利用率接近100%。

但如果是跨度较大且不连续的内存访问：

```
thread 0 → addr + 0
thread 1 → addr + 64
thread 2 → addr + 128
...
```

这不得不需要32次内存事务。

向量化访存可以让一个线程用一次指令加载多个连续元素（如 4×float）， 以更高的内存带宽效率、更少的指令数把数据搬进来，这样就可以减少指令数量还能利用内存事务。

先定义两个宏：

```c
// Row-major linear index
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Vectorized float4 load/store helper
#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
```

`OFFSET`是计算行主序矩阵的偏移，`FETCH_FLOAT4`按一个 `float4` 整体，拷贝到 `ldg_a_reg[reg_idx ... reg_idx+3]`，通常会生成 1 条 128-bit load 指令，使用更宽的load指令可以减少数据搬运次数。

如下是两段等价的代码：

```c
// 一次取一个FLOAT4
FETCH_FLOAT4(ldg_a_reg[reg_idx]) = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, a_tile_col, K)]);

// 等价于一次执行如下代码
ldg_a_reg[reg_idx + 0] = A_block[(a_tile_row + i) * K + (a_tile_col + 0)];
ldg_a_reg[reg_idx + 1] = A_block[(a_tile_row + i) * K + (a_tile_col + 1)];
ldg_a_reg[reg_idx + 2] = A_block[(a_tile_row + i) * K + (a_tile_col + 2)];
ldg_a_reg[reg_idx + 3] = A_block[(a_tile_row + i) * K + (a_tile_col + 3)];
```

于是，这一版的代码对于上一版本的优化主要体现在使用更宽的load指令访问内存，每个线程在同样的时间开销下可以搬运更多的数据，可以进一步提升性能。

直接在上一版代码的基础上修改代码：

```c
// 每个线程要搬运多少次
const int ldg_a_num = BK * BM / num_threads / 4; // 从A中搬运的次数
const int ldg_b_num = BK * BN / num_threads / 4; // 从B中搬运的次数
```

因为现在数据宽度变为4倍，所以下面的索引要重新计算：

```c
/* ------------------------------
 * Thread mapping for loading A tile
 * ------------------------------ */
const int a_tile_row = threadIdx.x / (BK / 4);
const int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
const int a_tile_stride = BM / ldg_a_num;
const int b_tile_row = threadIdx.x / (BN / 4);
const int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
const int b_tile_stride = BK / ldg_b_num;
```

数据加载如下，可以看到我们用`FETCH_FLOAT4`进行读取，这样一次load可以读到4个float宽度的数据。

```c
#pragma unroll
    for (int k = 0; k < K; k += BK) {
        /* ------------------------------
         * Load A tile into shared memory
         * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int reg_idx = (i / a_tile_stride) * 4;

            // load float4 from global A
            // 这里使用了寄存器进行过渡
            FETCH_FLOAT4(ldg_a_reg[reg_idx]) = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, a_tile_col, K)]);

            // store into shared As as transposed
            As[OFFSET(a_tile_col + 0, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 0];
            As[OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 1];
            As[OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 2];
            As[OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 3];
        }

        /* ------------------------------
         * Load B tile into shared memory
         * ------------------------------ */
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)])
                = FETCH_FLOAT4(B_block[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }

        __syncthreads(); // Ensure As and Bs are fully loaded

        // Advance global pointers
        A_block += BK;
        B_block += BK * N;

		....
    }
```

要注意的是，这里的A是转置后存到共享内存中，为什么要这样做呢？

这不得不联系到GPU共享内存本身的特性，共享内存被划分成32个Bank，每个Bank有4B的容量，当有**多个线程同时访问一个Bank的不同地址**时，就会出现Bank Conflict，那么这些访问请求会被拆分为顺序请求（一次访问被拆成多次），自然要更多的时间；反直觉的是，多个线程同时访问一个Bank的相同地址并不会出现Conflict，此时会触发广播机制，访问请求不会被拆分成顺序请求。

更具体的是，对于两个线程分别访问地址X和地址Y，且X不等于Y，但(x / 4) mod 32 等于 (Y / 4) mod 32那么就会出现Conflict，如下图所示，三种访问情况都没有Conflict，即使中间的5号Bank的一个地址被多个线程访问也没出现Conflict，因为存在广播机制。

![Irregular Shared Memory Accesses.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/examples-of-irregular-shared-memory-accesses.png)

但是下面这张图的中间部分是存在Conflict的，因为一个Bank中的不同地址被多个线程访问。

![Strided Shared Memory Accesses in 32 bit bank size mode.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/examples-of-strided-shared-memory-accesses.png)

更多信息可以参考：[NVIDIA DOC | Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=bank#shared-memory-5-x)

回顾V2的访问方式：

```c
#pragma unroll
    for (int t = 0; t < BK; ++t) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) { accum[i][j] += As[(ty + i) * BK + t] * Bs[t * BN + (tx + j)]; }
        }
    }
```

如果一个warp内的线程的 `((ty + i) * BK + t) mod 32`是一样的，则发生Bank Conflict，那么这就影响了执行速度，转置是为了避免这种问题。

我们写出最后的计算过程：

```c
/* ------------------------------
 * Compute: register-level GEMM
 * ------------------------------ */
#pragma unroll
for (int t = 0; t < BK; ++t) {
#pragma unroll
    for (int m = 0; m < TM; m += 4) { FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(t, ty + m, BM)]); }
#pragma unroll
    for (int n = 0; n < TN; n += 4) { FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(t, tx + n, BN)]); }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) { accum[i][j] += a_frag[i] * b_frag[j]; }
    }
}
```

最后写回到C Block：

```c
// Write back C
#pragma unroll
for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
        float4 c4 = FETCH_FLOAT4(C_block[OFFSET(ty + m, tx + n, N)]);
        c4.x = alpha * accum[m][n + 0] + beta * c4.x;
        c4.y = alpha * accum[m][n + 1] + beta * c4.y;
        c4.z = alpha * accum[m][n + 2] + beta * c4.z;
        c4.w = alpha * accum[m][n + 3] + beta * c4.w;
        FETCH_FLOAT4(C_block[OFFSET(ty + m, tx + n, N)]) = c4;
    }
}
```

分析一下性能：

![matmul_v3](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_7.png)

可以看到执行时间又进一步缩短了。

## V4 双缓冲流水线

源代码：[matmul_v4.cu](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/src/matmul_v4.cu)

在这个版本中，将继续针对内存访问进行优化。

在上一个版本中，我们每轮总是：

1. 从全局内存中读取数据到共享内存
2. 从共享内存读取数据进行计算

可见，这两个步骤都要使用共享内存，因此访存和计算基本是串行执行（难以重叠），我们能不能将访存和计算重叠，让两者并行执行？

答案是显然可以的，我们使用双缓冲策略实现显存块的交替使用，用两份缓冲区交替工作，一份用于"当前计算"，另一份提前"预取下一批数据"。注意，不是同时使用两块共享内存，而是任意时刻，每个线程只在读一块共享内存，只在写另一块共享内存，没有"同一数据被同时读写"。

双缓冲的作用是让『下一轮所需数据的准备』不必等『当前轮的计算』结束，通过两份互斥使用的缓冲区，把准备阶段和计算阶段在时间上重叠起来，从而隐藏内存和同步带来的延迟。本质上是用计算去覆盖访存这部分时间，即隐藏访问延迟。

像这样开辟双缓冲区：

```c
/* ------------------------------
 * Shared double buffer for A and B tiles
 * ------------------------------ */
__shared__ float As[2][BK * BM];
__shared__ float Bs[2][BK * BN];
```

同时，寄存器也要使用双缓冲：

```c
// register fragments
float a_frag[2][TM];
float b_frag[2][TN];
```

在流水线的启动状态，现在先将A和B的第1个Tile放到共享内存中，再从共享内存中搬到寄存器中：

```c
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        const int reg_idx = (i / a_tile_stride) * 4;
        FETCH_FLOAT4(ldg_a_reg[reg_idx]) = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, a_tile_col, K)]);

        // store A into shared as transposed
        As[0][OFFSET(a_tile_col + 0, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 0];
        As[0][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 1];
        As[0][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 2];
        As[0][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)])
            = FETCH_FLOAT4(B_block[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }

    __syncthreads();

    // preload frag for t=0 from shared buffer 0
#pragma unroll
    for (int m = 0; m < TM; m += 4) { FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]); }
#pragma unroll
    for (int n = 0; n < TN; n += 4) { FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]); }
```

这是第一个数据加载的步骤，先将数据从全局内存加载到`As`和`Bs`的`buffer 0`中，再将数据从`buffer 0`搬迁到寄存器的`buffer 0`中。

接着进入主循环执行流水线，在下面的主循环中，执行如下步骤：

```c
while (还有 K-tile) {

    // (1) 预取下一tile（global → reg）
    // 将next tile放到寄存器，等到合适窗口再写入 shared
    // 避免在 compute 阶段直接写 shared 造成同步/争用
    prefetch_next_tile_to_registers()

    // (2) 计算当前 tile 的前 BK-1 拍
    //     每一拍：load(k+1) + compute(k)
    for k = 0 .. BK-2:
        preload_frag_from_shared(k+1) // (shared → frag)
        compute_frag(k)
            
    // (3) 提交下一tile（reg → shared）并同步
    write_next_tile_to_shared()
    syncthreads()

    // (4) 计算当前 tile 的最后一拍
    compute_frag(BK-1)

    // 切换 shared buffer，推进 K
}
```

在V3版本的代码中，每次从全局内存中读取A的数据要分为两步：从全局内存搬到寄存器，再从寄存器搬到共享内存。在上述的循环中，我们其实将这两步分为了(1)和(3)。

值的注意的是，`Compute BK-1`被单独拿出来执行是因为`BK-1`已经是最后一块，不存在预取下一块，所以这一段被拿到最后执行，正好拿来覆盖内存访问。

在V3中，每轮迭代是按照1-3-2-4的顺序进行的，由于都要使用共享内存，所以这4个步骤只能串行执行。那能不能按照1-2-4-3的顺序执行？也是可以的，这样流程就变成了：

```c
while (还有 K-tile) {

    // (1) 预取下一tile（global → reg）
    prefetch_next_tile_to_registers()

    // (2) 计算当前 tile 的前 BK-1 拍
    //     每一拍：load(k+1) + compute(k)
    for k = 0 .. BK-2:
        preload_frag_from_shared(k+1)
        compute_frag(k)

    // (4) 计算当前 tile 的最后一拍
    compute_frag(BK-1)
    
    // (3) 提交下一tile（reg → shared）并同步
    write_next_tile_to_shared()
    syncthreads()

    // 切换 shared buffer，推进 K
}
```

如果这样的话，执行时间会稍微增加，因为`syncthreads()`指令被延迟，即同步点被推迟，导致下一轮启动被推迟。

可以总结：

- (1)和(2)可以重叠，不争用共享内存
- (2)内部的`preload_frag_from_shared(k+1)`和`compute_frag(k)`可以重叠
- (3)和(4)可以重叠，不争用共享内存
- (1)和(3)不能重叠，因为访问同一批寄存器

于是，整个循环中有两个级别的流水线：

- Tile级流水线：取 next tile 和计算 current tile 可以重叠
- Fragment级流水线：在计算 current tile 中，compute 和 load 可以重叠

如下写出完整代码：

```c
// ---------------------------
// Main loop over K tiles
// ---------------------------
do {
    const int next_k = k_base + BK;

    // prefetch next tile from global into registers
    if (next_k < K) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int reg_idx = (i / a_tile_stride) * 4;

            FETCH_FLOAT4(ldg_a_reg[reg_idx])
                = FETCH_FLOAT4(A_block[OFFSET(a_tile_row + i, next_k + a_tile_col, K)]);
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int reg_idx = (i / b_tile_stride) * 4;
            FETCH_FLOAT4(ldg_b_reg[reg_idx])
                = FETCH_FLOAT4(B_block[OFFSET(next_k + b_tile_row + i, b_tile_col, N)]);
        }
    }

    // shared buffer we are reading (current tile)
    // 这是一个状态变量，标记从哪块Shared Memory加载数据，在下一轮会翻转
    const int load_index = write_index ^ 1; 

#pragma unroll
    for (int t = 0; t < BK - 1; ++t) {
        // load next k-frag while computing current frag
#pragma unroll
        for (int m = 0; m < TM; m += 4) {
            FETCH_FLOAT4(a_frag[(t + 1) & 1][m]) = FETCH_FLOAT4(As[load_index][OFFSET(t + 1, ty + m, BM)]);
        }
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[(t + 1) & 1][n]) = FETCH_FLOAT4(Bs[load_index][OFFSET(t + 1, tx + n, BN)]);
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) { accum[i][j] += a_frag[t & 1][i] * b_frag[t & 1][j]; }
        }
    }

    if (next_k < K) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int reg_idx = (i / a_tile_stride) * 4;

            As[write_index][OFFSET(a_tile_col + 0, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 0];
            As[write_index][OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 1];
            As[write_index][OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 2];
            As[write_index][OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[reg_idx + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int reg_idx = (i / b_tile_stride) * 4;

            FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)])
                = FETCH_FLOAT4(ldg_b_reg[reg_idx]);
        }

        __syncthreads();

        // preload frag for next tile's t=0
#pragma unroll
        for (int m = 0; m < TM; m += 4) {
            FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_index][OFFSET(0, ty + m, BM)]);
        }
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, tx + n, BN)]);
        }

        write_index ^= 1; // 翻转状态，切换缓冲区，使得每轮交替使用缓冲区
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) { accum[i][j] += a_frag[(BK - 1) & 1][i] * b_frag[(BK - 1) & 1][j]; }
    }

    k_base = next_k; // 向前移动
} while (k_base < K);
```

如下是性能分析结果，可见执行时间进一步缩短：

![matmul_v4](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_8.png)

## V5 Warp Tiling

源代码：[matmul_v5.cu](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/src/matmul_v5.cu)

在上述4个版本中，都只使用了Block Tiling进行优化，也就是用一个线程块处理一个矩阵Block，但事实上，GPU的最小调度单位是warp，使用Warp Tiling优化，在更细粒度下控制线程对内存的访问，避免Bank Conflict的发生。另外，一个warp中的线程是天然同步的，每时每刻执行同一条指令，不需要任何显式同步原语，也消除大量不必要的 `__syncthreads()`，降低了性能开销。

我们将连续的数据范围分配给同一个 warp，可以使 warp 内 32 个线程在执行同一条内存指令时访问地址单调、紧凑且覆盖范围最小，从而减少内存事务数量并提高 cache line 的有效利用率；相比之下，仅在 block 粒度保证数据连续并不能确保 warp 内访问模式满足这些条件。

一个线程块还是算一个BM × BN大小的块，一个warp算其中更小的WM × WN大小的块:

```c
// block tile coords on C
const int block_col = blockIdx.x; // along N
const int block_row = blockIdx.y; // along M

// warps in this block
const int warp_idx = threadIdx.x / WARP_SIZE;
const int lane = threadIdx.x % WARP_SIZE;

constexpr int WARPS_PER_COL = BN / WN;
constexpr int WARPS_PER_ROW = BM / WM;

// (warp_row, warp_col) within block tile
const int warp_col = warp_idx % WARPS_PER_COL;
const int warp_row = warp_idx / WARPS_PER_COL;
```

同时，一个warp还要切分出更小的块，因为每个 thread 的寄存器能力有限，无法一次铺满 WM×WN大小的块。于是一个warp 负责的 WM×WN 块并不是一次性算完，而是分多次算：

```
for wsr in [0, WMITER)
  for wsc in [0, WNITER)
    计算一个 WSUBM × WSUBN
```

warp内部细分：

```c
// Warp micro-tiling
// WMITER is derived: how many "sub-rows" we iterate in WM direction per warp step
// WSUBM/WSUBN define warp's internal subdivision
constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
constexpr int WSUBM = WM / WMITER; // 每个迭代处理的M方向子块大小
constexpr int WSUBN = WN / WNITER; // 每个迭代处理的N方向子块大小

// lane -> (thread_row_in_warp, thread_col_in_warp)
// warp covers WSUBM x WSUBN per (w_sub_row_idx, w_sub_col_idx) region,
// each thread computes TMxTN
constexpr int THREADS_PER_WSUBN = WSUBN / TN;

// warp内线程的二维位置
const int thread_col_in_warp = lane % THREADS_PER_WSUBN;
const int thread_row_in_warp = lane / THREADS_PER_WSUBN;
```

剩下的计算逻辑和上一版本基本类似，只不过将矩阵“切的更细”。

这是我们最终的优化版本，我们来看看性能测试报告：

![matmul_v5](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_9.png)

可见性能又有了显著的提升，再用同样的测试方法，和[CUTLASS](https://github.com/NVIDIA/cutlass)进行对比，如下是一段测试代码：

```c
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#define CHECK_CUDA_ERROR(call)                                                                 \
    do {                                                                                       \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

int main() {
    /* =============================
     * Matrix size
     * ============================= */
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    size_t bytes_C = size_t(M) * N * sizeof(float);

    /* =============================
     * Host allocation & init
     * ============================= */
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    /* =============================
     * Device allocation
     * ============================= */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    /* =============================
     * CUTLASS GEMM 定义（SIMT）
     * ============================= */
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,        
        cutlass::arch::Sm120,               
        cutlass::gemm::GemmShape<128,128,8>,   // Threadblock tile
        cutlass::gemm::GemmShape<64,64,8>,     // Warp tile
        cutlass::gemm::GemmShape<1,1,1>,       // Instruction tile (SIMT)
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 1, ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2   
    >;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args(
        problem_size,
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {alpha, beta}
    );

    /* =============================
     * Warmup
     * ============================= */
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed\n";
            return -1;
        }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* =============================
     * Timed run
     * ============================= */
    const int repeat_iters = 10;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < repeat_iters; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed\n";
            return -1;
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / repeat_iters;

    /* =============================
     * GFLOPS
     * ============================= */
    double flops = 2.0 * double(M) * N * K;
    double gflops = flops / (avg_ms * 1e6);

    printf("CUTLASS SGEMM (SIMT FP32):\n");
    printf("  Avg time: %.3f ms\n", avg_ms);
    printf("  Perf:     %.2f GFLOPS\n", gflops);

    /* =============================
     * Cleanup
     * ============================= */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return 0;
}
```

编译命令：

```c
nvcc -O3 -std=c++17 \
  -I$CUTLASS/include \
  -I$CUTLASS/tools/util/include \
  cutlass_sgemm_bench.cu \
  -o cutlass_sgemm_bench
```

查看性能测试报告：

![matmul_v5](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/img/matmul_10.png)

可以发现我们的手写kernel在这样的测试环境下的性能已经非常接近CUTLASS了😍

## 补充与总结

本文通过 **V1 → V5** 五个版本，逐步手写实现并优化一个 **CUDA SGEMM（FP32 矩阵乘法）Kernel**，目标是：

- 减少全局内存访问
- 提高数据复用
- 提升计算强度（Compute / Memory Ratio）
- 隐藏访存与同步延迟
- 最终性能逼近 **CUTLASS（SIMT 路径）**

可做如下总结：

| 版本                           | 核心思想                      | 主要做法                                                     | 优点                                                         | 主要问题或局限                                               | 效果与结论                                      |
| ------------------------------ | ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| V1 最朴素实现（Baseline）      | 一线程一元素                  | 每个线程计算一个 C[m,n] 元素；从全局内存读取 A 的一整行与 B 的一整列；在寄存器中完成累加 | 实现方式直观简单；便于理解 CUDA 线程与网格模型               | 全局内存访问极其频繁；几乎没有数据复用；计算强度极低；L1 Cache 访问压力巨大 | 性能完全受限于内存带宽，仅能作为正确性验证      |
| V2 Thread Tile + Shared Memory | Block 与 Thread 分块          | 一个线程块计算一个 BM×BN 的 C Block；K 维按 BK 切分；每个线程计算 TM×TN 子块 | 引入共享内存；线程块内实现数据复用；显著减少全局内存访问；提高寄存器级计算密度 | 参数选择复杂；索引计算与线程映射难度较高                     | 执行时间明显下降，计算密度和缓存利用率显著提升  |
| V3 向量化访存                  | 合并访存与 Bank Conflict 规避 | 使用 float4 进行向量化加载与存储；A Tile 在共享内存中以转置形式存放 | 减少访存指令数量；提高内存事务利用率；有效降低共享内存 Bank Conflict | 对数据对齐与边界条件要求更严格                               | 内存吞吐进一步提升，Kernel 执行时间继续缩短     |
| V4 双缓冲流水线                | 延迟隐藏（Latency Hiding）    | 使用两套共享内存缓冲区交替工作；预取下一 Tile 的同时计算当前 Tile；构建 Tile 级与 Fragment 级流水线 | 实现访存与计算重叠；有效隐藏全局内存与同步延迟               | 代码结构复杂度显著提升，可读性与维护成本增加                 | 执行时间进一步下降，SM 利用率明显提高           |
| V5 Warp Tiling（最终形态）     | 以 Warp 为基本计算单元        | 将 Block Tile 细化为 Warp Tile；每个 Warp 负责 WM×WN 子块；Warp 内进行 micro-tiling | Warp 内天然同步；显著减少同步开销；内存访问模式更加规整；几乎消除 Warp 内 Bank Conflict | 实现难度高；参数设计与调优复杂                               | 性能接近 CUTLASS 的 SIMT 实现，达到准工业级水平 |
