# PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference

> Krishna Teja Chitty-Venkata, Jie Ye, Xian-He Sun, Anthony Kougkas, Murali Emani, Venkatram Vishwanath, Bogdan Nicolae

![111](fig1.png)

## Abstract

KV caching significantly improves the efficiency of Large Language Model
(LLM) inference by storing attention states from previously processed tokens,
enabling faster generation of subsequent tokens. However, as sequence length
increases, the KV cache quickly becomes a major memory bottleneck. To address
this, we propose PagedEviction, a novel fine-grained, structured KV cache
pruning strategy that enhances the memory efficiency of vLLM's PagedAttention.
Unlike existing approaches that rely on attention-based token importance or
evict tokens across different vLLM pages, PagedEviction introduces an efficient
block-wise eviction algorithm tailored for paged memory layouts. Our method
integrates seamlessly with PagedAttention without requiring any modifications
to its CUDA attention kernels. We evaluate PagedEviction across
Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, and Llama-3.2-3B-Instruct models
on the LongBench benchmark suite, demonstrating improved memory usage with
better accuracy than baselines on long context tasks.

- 使用KV 的静态L2 Norm信息估计他们的重要性
- Prefill时采用token-wise evication策略，然后再划分block 存储
- Decode时，如果新的token填满一个block时，则evict最不重要的一个block
- 实验部分没有和quest比较，精度下降3–5%