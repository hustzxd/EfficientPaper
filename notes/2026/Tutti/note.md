# Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving

> Shi Qiu, Yifan Hu, Xintao Wang, Wenhao Zhu, Jianqin Yan, Hao Chen, Kaiqiang Xu, Kai Chen, Yiming Zhang

![111](cover.jpg)

## Abstract

LLM serving relies on prefix caching to improve inference performance. As growing contexts push key-value (KV) cache footprint far beyond GPU HBM and CPU DRAM capacity, KV cache is increasingly offloaded to NVMe SSDs. Unfortunately, restoring KV cache from SSDs suffers from poor I/O performance and incurs significant GPU stalls. This is primarily because the fragmented GPU memory layout results in a massive number of tiny random I/Os, rendering the low-parallelism CPU a severe bottleneck even with GPU Direct Storage (GDS), which still relies on CPU intervention to initiate each I/O and thus remains CPU-centric. This paper presents Tutti, an efficient SSD-backed KV caching solution that eliminates CPU intervention from the critical data and I/O control paths between HBM and SSDs. At the core of Tutti is a GPU-centric KV cache object store, in which the CPU is only responsible for asynchronously loading I/O kernels once per layer to the GPU. Tutti saturates NVMe SSD bandwidth and reduces GPU stalls to near zero through the following designs: (i) we provide a GPU-native object abstraction that enables bulk KV cache transfers and management; (ii) we re-architect the GPU storage stack by introducing GPU io_uring to support asynchronous GPU direct object I/O; and (iii) we propose slack-aware I/O scheduling to avoid GPU resource contention. We have implemented Tutti and integrated it to vLLM. Extensive evaluation shows that compared to the state-of-the-art GDS-enabled, SSD-backed LMCache, Tutti reduces TTFT by 78.3% under strict SLO constraints and improves the achievable request rate by 2x. The serving cost is reduced by 27%. Tutti achieves nearly the same inference performance as DRAM-backed LMCache, while providing almost infinite capacity.


---

*以下总结由 MiMo 生成：*

这篇论文针对长上下文大语言模型服务中KV缓存容量不足的问题，提出了一种基于SSD的高效缓存方案。研究团队设计了Tutti系统，通过GPU中心化的对象存储和异步I/O机制，消除了CPU在关键数据路径上的干预。实验表明，Tutti在严格SLO约束下将首token延迟降低了78.3%，请求吞吐量提升2倍，服务成本降低27%，同时实现了接近DRAM缓存的推理性能。

---

## Tutti vs Strata：GPU-centric I/O 的两种路径

两者都以 GPU 为中心优化 KV cache 的数据搬运，但解决的问题层级不同：

### Strata 的 GPU 辅助 I/O

- **数据源**：CPU DRAM → GPU HBM
- **瓶颈**：PCIe 带宽利用率低（碎片化小页面传输，仅利用 22% 理论带宽）
- **机制**：用 CUDA kernel 替代 cudaMemcpyAsync，GPU 线程直接从 CPU pinned memory 读数据到寄存器再写到 HBM
- **参数**：2 个 CUDA block × 1024 线程，~50 GB/s，<5% 计算干扰
- **附带收益**：传输过程中零开销做 layout 变换（layer-first ↔ page-first）
- **本质**：GPU 线程接管搬运工作，用 GPU 大规模并行解决小粒度传输的并发度不足

### Tutti 的 GPU-Centric SSD Object Store

- **数据源**：SSD → GPU HBM（绕过 CPU）
- **瓶颈**：CPU 在 SSD restore path 上成为瓶颈（即使 GDS 也需要 CPU 发起每个 I/O）
- **机制**：GPU-native 对象抽象 + GPU io_uring 异步直接 I/O + slack-aware I/O 调度
- **关键设计**：CPU 仅负责每层异步加载一次 I/O kernel，之后完全由 GPU 线程管理 SSD 数据路径
- **本质**：把 SSD→CPU→GPU 两级路径变成 GPU 直接管理 SSD 数据，消除 CPU 中间人

### 互补关系

```
SSD ──(Tutti: GPU 直管 SSD, 绕过 CPU)──→ CPU DRAM ──(Strata: GPU 辅助搬运)──→ GPU HBM
```

在完整的分层缓存体系中，Tutti 优化 SSD→CPU 这一段（存储层），Strata 优化 CPU→GPU 这一段（传输层）。两者结合可实现从 SSD 到 GPU HBM 的全链路高效数据路径。
