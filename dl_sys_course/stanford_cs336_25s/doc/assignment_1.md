# CS336 ASIGNMENT 1 - BASICS

题目链接：https://github.com/stanford-cs336/assignment1-basics.git

在该实验中，要几乎从0实现并训练一个标准Transformer模型，这包含：

- BPE Tokenizer
- Transformer Language Model
- 交叉熵损失函数 & AdamW优化器
- 训练主流程

初始化环境和下载数据集的方法写在了README中，要实现的代码在`adapters.py`中，完善所有包含`NotImplementedError`的函数。

## BPE Tokenizer

tokenizer也叫作分词器，作用就是把一个句子分成一个个token，而BPE的目标就是自动学习一套高频subword作为token，其思想就是不断合并语料中出现频率最高的相邻符号对，直到达到预设词表大小。

合并规则如下，首先是输入一个文本：

```
lower
```

初始切分（通常每个词末尾加特殊符号 `</w>`）：

```
l o w e r </w>
```

根据训练阶段得到的 merge rules（有顺序）：

```
(l, o) → lo
(lo, w) → low
(e, r) → er
```

合并后：

```
low er </w>
```

输出 token ids：

```
["low", "er"]
```

训练方法如下。

输入：

- 大规模文本语料
- 目标词表大小（如 30k/50k）

**Step 1：初始化词表（字符级）**

语料：

```
low lower lowest
```

统计词频并拆成字符：

```
l o w </w>        (low)
l o w e r </w>    (lower)
l o w e s t </w>  (lowest)
```

初始 token 集合 = 所有字符 + `</w>`

**Step 2：统计所有相邻 token 对的频率**

例如：

```
(l, o): 3
(o, w): 3
(w, e): 2
(e, r): 1
(e, s): 1
```

Step 3：选择频率最高的 token 对进行合并

```
(l, o) → lo
```

更新语料：

```
lo w </w>
lo w e r </w>
lo w e s t </w>
```

**Step 4：重复步骤 2 & 3**

下一轮：

```
(lo, w) → low
```

直到：

- 达到词表大小上限
- 或没有可合并的 token 对

**Step 5：保存结果**

最终得到：

- **subword 词表**
- **有序 merge rules 列表**

