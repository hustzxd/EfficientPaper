# FlatCake: A Flat KVCache Storage Architecture for LLM Serving

> Xin Huang, Teng Ma, Zhuoran Li, Yibo Zhao, Zhaoshi Li, Qianchao Zhu, Wei Yang, Chenxi Wang, Tao Xie, Youwei Zhuo

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> 来源：[arXiv:2610.38355v1](https://arxiv.org/abs/2610.38355v1)，2026-10-28。已阅读 14 页正文及附录；实验为作者报告，未在本地复现。

## Abstract

KV cache reuse accelerates LLM serving, but existing systems manage host DRAM and NVMe SSD as a tiered hierarchy where DRAM acts as an intermediate cache. FlatCake exposes host DRAM and NVMe SSD as peer media in a unified 64-bit address space. Because KV blocks are write-once, immutable, and reused at session scope, each block is placed once on its target medium in session-contiguous, layer-major order and never migrates. FC-MAS separates DRAM-ready and SSD-preparing queues, while FC-CIO streams coalesced SSD reads directly into a bounded GPU window via GPUDirect Storage. On a production Claude Code trace with DeepSeek-V4-Flash, FlatCake sustains a 94% cache hit rate, reduces Mooncake's SSD read amplification by 6.5x-6.7x, and improves peak-load mean TTFT and total token throughput by 1.6x. On Llama-3.1-70B across 8K-128K contexts, it reduces mean TTFT by up to 2.4x over Mooncake and 1.8x over Mooncake-GDS.

## 一句话总结

FlatCake 利用 KV block 的写一次、不可变和 session 复用属性，把 DRAM 与 SSD 从“逐级迁移的缓存层次”改成同级 peer media，在创建时决定位置、按 session/layer-major 布局写盘，并用队列分离和 GDS 合并读取避免 DRAM staging 与 SSD 碎片化。

## 创新点

1. **FC-GVA：写入时一次定位。**统一 64-bit GVA 把 `MediaType`、`NodeID` 和 offset 编进地址，按共享 prompt、短对话和长 agentic session 等 reuse scope，直接将 block 放入 DRAM 或 SSD；之后不发生跨层 migration、dirty writeback 或逐页 LRU 迁移。SSD 空间以整个 session extent 回收，避免按 bucket 回收破坏其它会话的连续性。
2. **FC-MAS：按媒体把请求分成两条队列。**DRAM-ready 请求直接进入 GPU；SSD-preparing 请求按大小和等待时间的 HRRN 优先级预取，完成后按原始 arrival timestamp 重新进入 ready queue。这样快 DRAM 请求不会被多 GB SSD restore 阻塞，SSD session 也不会被持续 DRAM 流量饿死。
3. **FC-CIO：写时布局与读时合并协同。**同一 session 的 KV 按 layer-major 写进预分配连续 GPU File；读取时按 `(file_id, offset)` sort-merge，将每层许多 block 读合并为大范围 SSD 命令。GDS 直接把 layer extent 读入固定双缓冲 GPU window，再用 CUDA scatter 写入 SGLang 分页 KV slots，避免把完整 prefix 先放进 host staging buffer。

## 带来什么提升

1. **生产 Claude Code trace：**8×B200、DeepSeek-V4-Flash、98,827 requests、21.6B input tokens、393 sessions、0.1-1.0 QPS。FlatCake delivered cache hit 保持 **94.16%-94.57%**，接近 94.72% 的 trace reuse ceiling；Mooncake 从低负载 87.9% 降到 0.5 QPS 以上的 64.3%。峰值 1.0 QPS 时，FlatCake mean TTFT 7.57s、P50 6.26s、total throughput 98.3K tokens/s，相对 Mooncake 分别约 **1.6x、1.8x、1.6x**。
2. **SSD 路径效率：**FlatCake 每 worker SSD read bandwidth 约 4.96-5.12GB/s，Mooncake 约 2.69GB/s，约 **1.9x**；峰值 QPS 下 FlatCake prefetch latency 约 1.64s，Mooncake 9.50s，约 **5.8x** 差距。Mooncake 的 normalized SSD read amplification 为 FlatCake 的 **6.51x-6.72x**，FlatCake 的 write amplification 固定为 1.0x；Mooncake 因撤销 prefetch 和重新计算，write amplification 为 1.007x-1.018x。
3. **跨模型与上下文长度：**4×A100 上的 Llama-3.1-70B、8K-128K context 中，FlatCake mean TTFT 相对 Mooncake 最高 **2.4x**、相对 Mooncake-GDS 最高 **1.8x**；Mooncake-GDS 在 16K 接近 FlatCake，但 64K 后因 eviction-scattered SSD layout 退化。实验说明 GDS 本身不能修复写入时碎片化。
4. **队列机制的独立贡献：**去掉 FC-MAS 的单 FCFS 队列后，双队列使 mean TTFT 降 **10%-11%**、P99 降 **27%**、total token throughput 提升 **12%**；FC-CIO 的 layer-major + sort-merge 则承担顺序 SSD 传输和 GDS streaming 的主要收益。

## 备注

- **适用工作集：**FlatCake 针对双峰 reuse：大量 one-shot block 加上超出 DRAM 的多 GB agent session。若 working set 可留在 host DRAM、prefix reuse 很低、或请求没有长会话复用，SSD peer storage 和预分配写成本的收益会下降。论文没有把 94% 命中率推广到所有 LLM workload。
- **系统边界：**主实验采用 SGLang v0.5.19、Mooncake v0.3.13、CUDA 13.1、GDS 1.16.1、8×B200；Llama 扫频采用 4×A100。容量预算为 1TiB DRAM KV + 4TiB NVMe KV。对象存储、CXL、跨节点 NVMe-oF 是设计可扩展方向，未在主实验中验证。
- **成本与实现：**FlatCake 约 8.8K 行 C++/CUDA 和 7.1K 行 Python，依赖预分配 GPU File、hugepage/pinned DRAM、GDS 注册缓冲区和 session extent 回收。统一地址和写时 placement 减少 migration，但需要容量协调、SSD extent 管理以及对长期 session 的回收策略。
- **与现有系统：**它与 Mooncake、LMCache、HiCache 的层级 KV 设计形成直接对照；与 KVStream 的多 tier completion routing、Tutti 的 GPU-centric SSD、HiSparse 的 sparse-attention host cache 互补。FlatCake 的关键增量是 write-time placement 和 session-contiguous layer-major layout，而非只启用 GDS。
