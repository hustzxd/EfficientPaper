# vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention

![](fig5.png)

## Abstract

PagedAttention is a popular approach for dynamic memory allocation in LLM
serving systems. It enables on-demand allocation of GPU memory to mitigate KV
cache fragmentation -- a phenomenon that crippled the batch size (and
consequently throughput) in prior systems. However, in trying to allocate
physical memory at runtime, PagedAttention ends up changing the virtual memory
layout of the KV cache from contiguous to non-contiguous. Such a design leads
to non-trivial programming and performance overheads.
  We present vAttention -- an approach that mitigates fragmentation in physical
memory while retaining the contiguity of KV cache in virtual memory. We achieve
this by decoupling the allocation of virtual and physical memory using CUDA
virtual memory management APIs. We also introduce various LLM-specific
optimizations to address the limitations of CUDA virtual memory support.
Overall, vAttention is a simpler, portable, and performant alternative to
PagedAttention: it supports various attention kernels out-of-the-box and
improves LLM serving throughput by up to 1.23x compared to the use of
PagedAttention-based kernels of FlashAttention and FlashInfer.

Paged Attention的改进

Paged Attention的缺点

- Requires Re-writing the Attention Kernel
    - KV Cache不是连续存储的，需要重写kernel
- Adds Redundancy in the Serving Framework
    - 在runtime时，需要同时访问多个块的kv cache，但是属于virtual memory的不同地址。Virtual Memory实际上通过操作系统找到

Physical Memory有一个相似的操作。

- Performance Overhead
    - Runtime overhead on the GPU
        - 多了一些运算逻辑，导致GPU上有额外的时间开销
    - Runtime overhead on the CPU

vAttention 发现两个现象

- KV cache memory requirement is predictable on a per-iteration basis.
    - 在decoding时，每次增长的memory是可以提前预测的
- KV cache does not require high memory allocation bandwidth.
    - 统计发现KV Cache需要Memory最高只有750MB/s

**在系统层面进行分页，保证在virtual memory上是连续的，而不是PagedAttention在用户空间中在Virtual Memory上进行分页。**