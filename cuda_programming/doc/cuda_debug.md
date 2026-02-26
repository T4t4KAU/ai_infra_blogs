# CUDA程序的调试方法

本文展示在vscode中如何调试CUDA程序

在当前目录下实现一个vec_add.cu，完成一个简单的向量加法：

```c++
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                   cudaGetErrorString(_e), (int)_e);                          \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__global__ void vecAddKernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static void init(float* x, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        x[i] = scale * (0.001f * (float)i) + 1.0f;
    }
}

static bool check(const float* a, const float* b, const float* c, int n) {
    int bad = 0;

    for (int i = 0; i < n; ++i) {
        float ref = a[i] + b[i];
        float err = std::fabs(c[i] - ref);
        if (err > 1e-5f) {
            if (bad < 10) {
                std::fprintf(stderr, "Mismatch at %d: c=%f ref=%f (a=%f b=%f)\n",
                            i, c[i], ref, a[i], b[i]);
                }
            ++bad;
        }
    }
    
    if (bad) {
        std::fprintf(stderr, "FAILED: %d mismatches\n", bad);
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);  // 默认 1M 元素
    if (n <= 0) n = 1 << 20;

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Using GPU %d: %s\n", device, prop.name);
    std::printf("n = %d\n", n);
    
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float*)std::malloc(bytes);
    float *h_b = (float*)std::malloc(bytes);
    float *h_c = (float*)std::malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    init(h_a, n, 1.0f);
    init(h_b, n, 2.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    std::printf("Launching vecAddKernel<<<%d, %d>>>\n", blocks, threads);
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = check(h_a, h_b, h_c, n);
    std::printf("Result: %s\n", ok ? "OK" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    CUDA_CHECK(cudaDeviceReset());

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
 
    return 0;
}
```

编写CMakeList.txt用于编译程序：

```cmake
cmake_minimum_required(VERSION 3.20)
project(cuda_test LANGUAGES CXX CUDA)

add_executable(vec_add vec_add.cu)

# 推荐：指定 CUDA 标准
set_target_properties(vec_add PROPERTIES
  CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED ON
)

# Debug 时开启 device 调试信息（-G）+ 主机调试信息（-g）
# Release 时不启用（避免性能损失）
target_compile_options(vec_add PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g>
)

# 可选：开启更友好的警告（按需）
target_compile_options(vec_add PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# 现代 CMake：链接 CUDA runtime（可选，但推荐写上更明确）
find_package(CUDAToolkit REQUIRED)
target_link_libraries(vec_add PRIVATE CUDA::cudart)
```

执行编译命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

在.vscode文件夹中添加launch.json：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug CUDA (cuda-gdb)",
      "type": "cuda-gdb",
      "request": "launch",
      "program": "${workspaceFolder}/build/vec_add",
      "cwd": "${workspaceFolder}",
      "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
      "args": [],
      "stopAtEntry": false,
      "setupCommands": [
        {
          "description": "Break on kernel launch",
          "text": "set cuda break_on_launch application",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
```

在vscode中打下断点，按F5开始调试