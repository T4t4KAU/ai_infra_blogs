# CUDA PROGRAMMING

本章节主要讲解CUDA编程基础，展现各种算子的优化技巧

|  名称  |                             路径                             |                             说明                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MATMUL | [matmul.md](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/cuda_programming/doc/matmul.md) | SGEMM算子的4种优化姿势，最终实现了一个性能非常接近CUTLASS的手写Kernel |

本章节的目录结构如下：

```
.
├── doc # 存放文档
├── img # 图片
├── scripts # 脚本文件
└── src # 源代码
```

所有代码都可以直接编译，例如：

```powershell
cd src # 进入源代码目录
make matmul_v1 # 编译matmul_v1.cu
make matmul_v1_ncu # 生成NCU报告
```