# How to Deploy the GLM LLM on Muxi Cluster

在之前已经讲过如何在沐曦单机上部署千问模型：[How to Deploy the Qwen LLM on Muxi GPUs](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/misc/doc/deploy_qwen_on_muxi.md)

本文讲解如何用两台机器部署GLM模型，模型可以在此下载：https://www.modelscope.cn/models/metax-tech/GLM-4.7-W8A8

现在在两台机器上都执行：

```powershell
docker run -it \
--device=/dev/dri \
--device=/dev/htcd \
--group-add video \
--name glm-ray \
--network=host \
--security-opt seccomp=unconfined \
--security-opt apparmor=unconfined \
--shm-size 120gb \
--ulimit memlock=-1 \
-v /mnt/data/models/:/models \
cr.metax-tech.com/public-ai-release-wb/hpcc/vllm:hpcc.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64 \
/bin/bash
```

注意这里指定`--network=host`，意味着容器不再有"自己的网络"，而是直接使用宿主机的网络栈。

两台机器、容器内都添加环境变量（很重要）：

```bash
export HPCC_VISIBLE_DEVICE=0,1,2,3,4,5,6,7
export HCCL_IB_HCA=mlx5_0,mlx5_4
export HPCC_SMALL_PAGESIZE_ENABLE=1
export GLOO_SOCKET_IFNAME=bond0
export HCCL_SOCKET_IFNAME=bond0
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export TRITON_ENABLE_HPCC_CHAIN_DOT_OPT=1
export TRITON_ENABLE_HPCC_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export FUSED_RMSNORM_QUANT=True
export CUDA_GRAPH_DP_USE_SUM_BS=False
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

主节点的IP是10.118.17.69，另一个是从节点10.118.17.70，按顺序在两台机器上执行
先启动主节点：

```powershell
ray start --head \
  --node-ip-address=10.118.17.69 \
  --port=6379 \
  --num-gpus=8
```

再启动从节点：

```powershell
ray start \
  --address=10.118.17.69:6379 \
  --node-ip-address=10.118.17.70 \
  --num-gpus=8
```

看到"Ray runtime started."就大概率是成功了，等两台都启动成功后，直接执行：

```powershell
ray status
```

可以看到：

```
Total Usage:
 0.0/256.0 CPU
 0.0/16.0 GPU
 0B/1.74TiB memory
 0B/228.00GiB object_store_memory
```

信息表明，现在有16个GPU了。

接着，只在主节点上执行：

```powershell
python -m vllm.entrypoints.openai.api_server \
  --model /models/GLM-4.7-W8A8 \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

输出一堆信息后，如果看到：

```
INFO:     Started server process [538]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

那就是成功了。