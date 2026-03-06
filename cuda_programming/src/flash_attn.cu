// flash_attention_v2_demo.cu
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "util.h"

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err__ = (call);                                                                                    \
        if (err__ != cudaSuccess) {                                                                                    \
            std::printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

#ifdef DEBUG
#define DEBUG_BLOCK(stmt)                                                                                              \
    do { stmt } while (0)
#else
#define DEBUG_BLOCK(stmt)                                                                                              \
    do {                                                                                                               \
    } while (0)
#endif

using FP = float;

// Tile sizes along sequence dimension
static constexpr int Br = 2;  // rows of Q / O processed per block
static constexpr int Bc = 2;  // rows of K / V processed per iteration (tile)

// Test sizes
static constexpr int kTestSeqLen = 4;
static constexpr int kTestDim    = 4;


// ------------------------------------------------------------
// FlashAttention-style forward kernel (online softmax, tiled K/V)
// This is single-batch, single-head version.
// ------------------------------------------------------------
__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, int dim, FP sm_scale) {
    // Thread layout: (tx, ty) = (0..Bc-1, 0..Br-1)
    const int tx = threadIdx.x; // key tile row index within [0, Bc)
    const int ty = threadIdx.y; // query row index within [0, Br)

    // Global query row this thread-row is responsible for
    const int q_row = blockIdx.y * Br + ty;
    if (q_row >= seqlen) {
        return;
    }

    // Number of K/V tiles along sequence length
    const int num_kv_tiles = (seqlen + Bc - 1) / Bc;

    // Shared memory tiles
    extern __shared__ unsigned char smem[];
    FP *sQ = reinterpret_cast<FP *>(smem); // [Br * dim]
    FP *sK = sQ + Br * dim;                // [Bc * dim]
    FP *sV = sK + Bc * dim;                // [Bc * dim]
    FP *sO = sV + Bc * dim;                // [Br * dim] accumulator (numerator)
    FP *sQK = sO + Br * dim;               // [Br * Bc] scores for this tile
    FP *sP = sQK + Br * Bc;                // [Br * Bc] exp(score - newMax)
    FP *sMax = sP + Br * Bc;               // [Br]
    FP *sDenom = sMax + Br;                // [Br]
    FP *sTileMax = sDenom + Br;            // [Br] temp
    FP *sTileDenom = sTileMax + Br;        // [Br] temp

    // Helpers for indexing flattened shared arrays
    auto Q_sh = [&](int r, int c) -> FP & { return sQ[r * dim + c]; };
    auto K_sh = [&](int r, int c) -> FP & { return sK[r * dim + c]; };
    auto V_sh = [&](int r, int c) -> FP & { return sV[r * dim + c]; };
    auto O_sh = [&](int r, int c) -> FP & { return sO[r * dim + c]; };
    auto QK_sh = [&](int r, int c) -> FP & { return sQK[r * Bc + c]; };
    auto P_sh = [&](int r, int c) -> FP & { return sP[r * Bc + c]; };

    // ----------------------------------------------------------
    // 1) Load Q tile and init O accumulator / max / denom
    // Each thread loads multiple columns with stride Bc (col = tx + t*Bc)
    // ----------------------------------------------------------
    for (int col = tx; col < dim; col += Bc) {
        Q_sh(ty, col) = Q[q_row * dim + col];
        O_sh(ty, col) = 0.f; // numerator accumulator init
    }

    if (tx == 0) {
        sMax[ty] = -INFINITY;
        sDenom[ty] = 0.f;
    }
    __syncthreads();

    // ----------------------------------------------------------
    // 2) Loop over K/V tiles along sequence dimension
    // ----------------------------------------------------------
    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const int k_row = tile * Bc + tx; // global key row this tx corresponds to

        // 2.1) Load K/V tile row (k_row) into shared (or zero if out-of-range)
        // Cooperative load over dim using ty lanes with stride Br
        for (int col = ty; col < dim; col += Br) {
            if (k_row < seqlen) {
                K_sh(tx, col) = K[k_row * dim + col];
                V_sh(tx, col) = V[k_row * dim + col];
            } else {
                // Out-of-range: make this key contribute nothing
                K_sh(tx, col) = 0.f;
                V_sh(tx, col) = 0.f;
            }
        }
        __syncthreads();

        // 2.2) Compute score for this (q_row, k_row)
        // If k_row is out-of-range, set score to -inf to get exp=0
        FP score = -INFINITY;
        if (k_row < seqlen) {
            FP dot = 0.f;
            for (int d = 0; d < dim; ++d) { dot += Q_sh(ty, d) * K_sh(tx, d); }
            score = dot * sm_scale;
        }
        QK_sh(ty, tx) = score;
        __syncthreads();

        // 2.3) Compute per-row tile max and tile denom (done by tx==0 for each row)
        if (tx == 0) {
            FP tmax = -INFINITY;
            for (int kk = 0; kk < Bc; ++kk) { tmax = fmaxf(tmax, QK_sh(ty, kk)); }
            sTileMax[ty] = tmax;
        }
        __syncthreads();

        const FP oldMax = sMax[ty];
        const FP tileMax = sTileMax[ty];
        const FP newMax = fmaxf(oldMax, tileMax);

        // 2.4) Compute exp(score - newMax) for each element in this tile row
        FP e = 0.f;
        // exp(-inf) will underflow to 0, fine.
        e = expf(QK_sh(ty, tx) - newMax);
        P_sh(ty, tx) = e;
        __syncthreads();

        if (tx == 0) {
            FP tden = 0.f;
            for (int kk = 0; kk < Bc; ++kk) { tden += P_sh(ty, kk); }
            sTileDenom[ty] = tden;
        }
        __syncthreads();

        const FP tileDenom = sTileDenom[ty];

        // 2.5) Update running denom and numerator accumulator (O_sh)
        // rescaleOld = exp(oldMax - newMax)
        const FP rescaleOld = expf(oldMax - newMax);
        const FP newDenom = sDenom[ty] * rescaleOld + tileDenom;

        // Update numerator per output column: O = O*rescaleOld + sum_k P*V
        for (int col = tx; col < dim; col += Bc) {
            FP acc = O_sh(ty, col) * rescaleOld;
            for (int kk = 0; kk < Bc; ++kk) { acc += P_sh(ty, kk) * V_sh(kk, col); }
            O_sh(ty, col) = acc;
        }

        if (tx == 0) {
            sMax[ty] = newMax;
            sDenom[ty] = newDenom;
        }
        __syncthreads();
    }

    // ----------------------------------------------------------
    // 3) Final normalize and write back: O = numerator / denom
    // ----------------------------------------------------------
    const FP denom = sDenom[ty];
    for (int col = tx; col < dim; col += Bc) { O[q_row * dim + col] = O_sh(ty, col) / denom; }
}

// Host wrapper
static void flash_attention_v2_cuda(FP *Q, FP *K, FP *V, FP *O, int seqlen, int dim) {
    FP sm_scale = 1.f / std::sqrt(static_cast<FP>(dim));

    dim3 block(Bc, Br, 1);
    dim3 grid(1, (seqlen + Br - 1) / Br, 1);

    // Shared memory size:
    // sQ[Br*dim] + sK[Bc*dim] + sV[Bc*dim] + sO[Br*dim]
    // + sQK[Br*Bc] + sP[Br*Bc]
    // + sMax[Br] + sDenom[Br] + sTileMax[Br] + sTileDenom[Br]
    size_t shmem_bytes = 0;
    shmem_bytes += sizeof(FP) * (Br * dim); // sQ
    shmem_bytes += sizeof(FP) * (Bc * dim); // sK
    shmem_bytes += sizeof(FP) * (Bc * dim); // sV
    shmem_bytes += sizeof(FP) * (Br * dim); // sO
    shmem_bytes += sizeof(FP) * (Br * Bc);  // sQK
    shmem_bytes += sizeof(FP) * (Br * Bc);  // sP
    shmem_bytes += sizeof(FP) * (Br);       // sMax
    shmem_bytes += sizeof(FP) * (Br);       // sDenom
    shmem_bytes += sizeof(FP) * (Br);       // sTileMax
    shmem_bytes += sizeof(FP) * (Br);       // sTileDenom

    flash_attention_v2_kernel<<<grid, block, shmem_bytes>>>(Q, K, V, O, seqlen, dim, sm_scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    DEBUG_BLOCK(std::printf("== flash v2 O ==\n"); print_device_matrix(O, seqlen, dim););
}

// ------------------------------------------------------------
// Test harness (compares naive vs flash kernel)
// ------------------------------------------------------------
static void test_attention_once(int seqlen, int dim) {
    // Host buffers (fixed: use vector to avoid new[]/free bugs)
    std::vector<FP> hQ(seqlen * dim);
    std::vector<FP> hK(seqlen * dim);
    std::vector<FP> hV(seqlen * dim);
    std::vector<FP> hO_ref(seqlen * dim);
    std::vector<FP> hO_flash(seqlen * dim);

    // Initialize inputs
    for (int i = 0; i < seqlen * dim; ++i) {
        hQ[i] = static_cast<FP>(std::rand()) / RAND_MAX;
        hK[i] = static_cast<FP>(std::rand()) / RAND_MAX;
        hV[i] = static_cast<FP>(std::rand()) / RAND_MAX;
    }

    FP *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, sizeof(FP) * seqlen * dim));
    CUDA_CHECK(cudaMalloc(&dK, sizeof(FP) * seqlen * dim));
    CUDA_CHECK(cudaMalloc(&dV, sizeof(FP) * seqlen * dim));
    CUDA_CHECK(cudaMalloc(&dO, sizeof(FP) * seqlen * dim));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), sizeof(FP) * seqlen * dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), sizeof(FP) * seqlen * dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), sizeof(FP) * seqlen * dim, cudaMemcpyHostToDevice));

    // Flash-style
    flash_attention_v2_cuda(dQ, dK, dV, dO, seqlen, dim);

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
}

int main() {
    std::srand(0);

    // Run multiple epochs
    const int epochs = 10;
    for (int i = 0; i < epochs; ++i) { test_attention_once(kTestSeqLen, kTestDim); }
    return 0;
}