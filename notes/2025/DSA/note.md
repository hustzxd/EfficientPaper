# DSA

![111](fig1.png)

## Abstract

1. 基于MQA实现，同一个token对应的所有的query共享稀疏位置
2. 不同token的query可以选择topk的kv进行运算，选择粒度为token-wise，与NSA不同
3. 加速实现涉及到gather运算，从global memory gather到shared memory上
