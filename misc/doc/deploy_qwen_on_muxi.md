# How to Deploy the Qwen LLM on Muxi GPUs

本文将演示如何在国产沐曦卡上使用Docker部署Qwen-32B大模型

打开终端，执行：

```powershell
ht-smi
```

这个命令可查看显卡配置信息，我的机器输出：

```
ht-smi  version: 2.2.9

=================== Mars System Management Interface Log ===================
Timestamp                                         : Wed Dec 31 19:57:21 2025

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| HT-SMI 2.2.9                       Kernel Mode Driver Version: 3.4.4            |
| HPCC Version: 3.3.0.15             BIOS Version: 1.30.0.0                       |
|------------------+-----------------+---------------------+----------------------|
| Board       Name | GPU   Persist-M | Bus-id              | GPU-Util      sGPU-M |
| Pwr:Usage/Cap    | Temp       Perf | Memory-Usage        | GPU-State            |
|==================+=================+=====================+======================|
| 0      Mars X201 | 0           Off | 0000:4b:00.0        | 0%          Disabled |
| 38W / 350W       | 34C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 1      Mars X201 | 1           Off | 0000:4c:00.0        | 0%          Disabled |
| 37W / 350W       | 35C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 2      Mars X201 | 2           Off | 0000:4e:00.0        | 0%          Disabled |
| 36W / 350W       | 33C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 3      Mars X201 | 3           Off | 0000:4f:00.0        | 0%          Disabled |
| 43W / 350W       | 34C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 4      Mars X201 | 4           Off | 0000:cb:00.0        | 0%          Disabled |
| 36W / 350W       | 34C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 5      Mars X201 | 5           Off | 0000:cc:00.0        | 0%          Disabled |
| 37W / 350W       | 35C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 6      Mars X201 | 6           Off | 0000:ce:00.0        | 0%          Disabled |
| 37W / 350W       | 32C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 7      Mars X201 | 7           Off | 0000:cf:00.0        | 0%          Disabled |
| 44W / 350W       | 34C          P0 | 858/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+

End of Log
```

可以看到：

```
Kernel Mode Driver Version: 3.4.4
HPCC Version: 3.3.0.15
BIOS Version: 1.30.0.0
```

有三个关键信息：

- **Kernel Mode Driver Version 3.4.4**：内核态驱动版本

- **HPCC Version 3.3.0.15**：沐曦高性能计算栈（类似 CUDA/ROCm 的运行时组件）

- **BIOS Version 1.30.0.0**：显卡固件版本

```
Attached GPUs : 8
```

表明了当前节点共 **8 张 Mars X201 显卡**

```
| 0 Mars X201 | 0 Off | 0000:4b:00.0 | 0% Disabled |
| 38W / 350W | 34C P0 | 858/65536 MiB | Available |
```

表明：

| 字段           | 含义                                         |
| -------------- | -------------------------------------------- |
| **Board Name** | Mars X201（显卡型号）                        |
| **GPU**        | 逻辑编号（0–7）                              |
| **Persist-M**  | 持久模式（Off，未开启）                      |
| **Bus-id**     | PCIe 地址                                    |
| **GPU-Util**   | GPU 利用率（0%）                             |
| **sGPU-M**     | 子 GPU 模式（Disabled，未启用 MIG 类似功能） |

以上是本机沐曦显卡的主要情况。

进入这个链接：https://developer.metax-tech.com/softnova/docker

下载所需要的Docker镜像：

![depoly_qwen_on_muxi_1](https://github.com/T4t4KAU/ai_infra_blogs/blob/main/misc/img/depoly_qwen_on_muxi_1.png)

选择"docker pull命令复制"，直接复制命令，粘贴到终端进行下载。注意尽量选择图中的image，其他的image可能会出问题。

```powershell
docker run -it \
--device=/dev/dri \
--device=/dev/htcd \
--group-add video \
--name Qwen3-32B \
--network=host \
--security-opt seccomp=unconfined \
--security-opt apparmor=unconfined \
--shm-size 100gb \
--ulimit memlock=-1 \
-v /path/to/models:/models \
cr.metax-tech.com/public-ai-release-wb/hpcc/vllm:hpcc.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64 \
/bin/bash
```

 注意，`-v /path/to/models:/models`是做目录挂载，是存放模型文件的目录，根据自己情况设置。

下载沐曦专用的Qwen-32B：

```powershell
cd /models
pip install modelscope
modelscope download --model metax-tech/Qwen3-32B.w8a8 --local_dir ./Qwen3-32B
```

下载完成后，直接执行：

```powershell
vllm serve /models/Qwen3-32B \
--port 8000 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.95 \
--max-model-len 2048 \
--dtype bfloat16
```

输出一堆信息后并等待一段时间后，可以看到终端最后输出：

```
INFO:     Started server process [17]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

这就表明启动成功了。

编写一段代码进行测试：

```python
import requests
import json

# Server address and API endpoint
# Assumed server IP: 10.118.17.101, port: 8000
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# HTTP headers
# Content-Type must be application/json
headers = {
    "Content-Type": "application/json"
}

# Request payload
# NOTE:
# The value of the "model" field must exactly match
# the actual model path used when starting the service
payload = {
    "model": "/models/Qwen3-32B",
    "messages": [
        {
            "role": "user",
            "content": "Introduce yourself"
        }
    ],
    "max_tokens": 64
}

# Send POST request to the inference service
response = requests.post(
    API_URL,
    headers=headers,
    data=json.dumps(payload),
    timeout=60
)

# Check HTTP status code
if response.status_code == 200:
    # Parse JSON response
    result = response.json()

    # Print full response (for debugging)
    print("Full response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Extract model reply (OpenAI-compatible format)
    if "choices" in result and len(result["choices"]) > 0:
        reply = result["choices"][0]["message"]["content"]
        print("\nModel reply:")
        print(reply)
else:
    # Print error information
    print(f"Request failed with status code {response.status_code}")
    print(response.text)

```

运行脚本后，发现模型输出：

```
<think>
Okay, I need to introduce myself to the user. First, I should mention my name and role clearly. I'm Qwen, a language model developed by Alibaba Cloud. Then, I should highlight my key features and capabilities, like multilingual support, dialogue understanding, and coding abilities.
```

至此启动成功。