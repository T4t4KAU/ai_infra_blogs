# LLAISYS ASSIGNMENT 1

提交链接：https://github.com/T4t4KAU/llaisys/commit/89d1ffef382588fd9cf29d1945243b3e486b2882

此处要求我们实现Tensor的基本功能，下面来逐一拆解。

## TASK 1.1 Tensor::load

这个函数的功能是将主机数据加载到张量上，可以完成从主机(CPU)到设备(GPU)的传输。

实现非常简单：

```c++
void Tensor::load(const void *src_) {
    std::memcpy(_storage->memory(), (std::byte *)src_, _storage->size());
}
```

拿到tensor内部存放数据的那个指针，然后执行一个内存拷贝就行。由于框架还没有支持GPU，所以只须用`std::memcpy`做一个简单的拷贝就行。

## TASK 1.2 Tensor::isContiguous()

这个函数的功能是检查张量的形状和步长，判断它在内存中是否连续。

首先搞清楚什么是形状(shape)和步长(stride)，代码中tensor的shape是一个数组，存tensor每个维度的长度，stride也是一个数组，存的是在某一维度上移动 1 个元素，需要跨越多少个元素个数。

shape 决定『逻辑结构』，stride 决定『内存跳跃规则』，两者一起决定元素如何映射到线性内存。

判断是否连续只要从后往前判断：

```c++
stride[i] == stride[i+1] * shape[i+1]
```

什么时候要内存连续呢？只要一个操作需要**按线性内存解释数据**，就必须连续；如果操作完全基于 stride 做索引计算，就不需要连续。

## TASK 1.3 Tensor::view

这个函数的功能是创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。要注意的是，新tensor的memory指针依然指向原tensor的内存，两个tensor是共享同一段内存的，这个通常称为原地操作。

只要根据新shape重新计算一下新的stride就行：

```c++
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    ptrdiff_t stride = 1;

    for (size_t i = 1; i <= ndim_; ++i) {
        new_strides[ndim_ - i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[ndim_ - i]);
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

要注意的是，索引`i`不要用`int`，因为`ndim_`是`size_t`，不同数据类型之间的比较在一些编译器上会报错。

## TASK 1.4 Tensor::permute

这个函数的功能是创建一个新张量，改变原始张量维度的顺序，也就是说调整一下shape数组的顺序就行，再同步改一下stride即可。

```c++
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());

    for (size_t i = 0; i < ndim(); i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

## TASK 1.5 Tensor::slice

创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。

```c++
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;

    const ptrdiff_t delta = static_cast<ptrdiff_t>(start) * _meta.strides[dim];
    const ptrdiff_t new_offset = static_cast<ptrdiff_t>(_offset) + delta;
    size_t dtype_size = utils::dsize(new_meta.dtype);

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage,
                                              static_cast<size_t>(new_offset) * dtype_size));
}
```

所谓切片就是修改一下shape，但是不用修改stride，因为内存映射没有变。

