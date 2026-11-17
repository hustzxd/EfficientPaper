# KVStream: Adaptive Parallel Storage Management for KV Caching in Large-Scale LLM Serving

> Avinash Maurya, Jaiaid Mobin, Jie Ye, M. Mustafa Rafique, Franck Cappello, Bogdan Nicolae

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> 来源：[arXiv:2610.31827v1](https://arxiv.org/abs/2610.31827v1)，2026-10-22。已阅读 17 页全文及附录；实验为作者报告，未在本地复现。

## Abstract

Long-context LLM serving for agentic, coding, and document workloads reuses prefixes whose key-value (KV) cache outgrows GPU and host memory, putting storage I/O on the critical path of the inference step for retrieving prefixes. A single node-local NVMe absorbs all retrievals, while the parallel file system or cloud file store that nodes already mount carries almost no inference traffic. We cast prefix retrieval as aggregate-bandwidth scheduling across these tiers and present KVStream, a KV cache engine that batches a step's reads, defers writes behind the forward pass, replicates each chunk on every tier, and lets each tier adaptively claim the next unread chunks once it is free. Our performance model shows this routing attains the aggregate-bandwidth bound without prediction or tuning, whereas any static split misses it by a margin that grows as per-tier bandwidths drift or fluctuate. It also predicts that the time-to-first-token (TTFT) gain grows with load, as faster retrieval drains the queue, and disappears when host memory or arrival order keeps reuse in cache. Plugged into LMCache and evaluated on four models, six datasets, and up to 128 GPUs, KVStream retrieves within 5% of the bound, delivers up to 1.49$\times$ higher offline throughput than LMCache, and lowers mean TTFT by up to 6.1$\times$ under bursty arrivals.

## 一句话总结

KVStream 将一步推理中所有未命中的 prefix KV 读取合并为一个 I/O 池，把每个 chunk 同时放在本地 NVMe 和并行文件系统上，并让先完成读取的 tier 继续领取下一批，从而在无需预测带宽的情况下动态利用两层存储的聚合带宽。

## 创新点

1. **将 prefix 回读从逐请求串行改为 step 级并行。**LMCache 依次读一个请求的全部 chunk、再拷贝到 GPU，之后才处理下一个请求，既让 NVMe 队列长期过浅，也让磁盘到主机与主机到 GPU 两条链路轮流空闲。KVStream 一次匹配整个 batch 的 KV chunk，把所有 storage-resident chunk 放进同一完成队列，任一 chunk 到达即可开始 H2D，并与其他读取重叠。缓冲区也按批分配，避免逐请求小块管理。
2. **写延后到下一次 forward pass。**新生成的 KV 先暂存在主机内存，读取阶段不提交写；等所有读取发完、forward 开始后，再让写与计算重叠，减少读写干扰。写并未消失：若 forward 太短或写缓冲回压，写仍会侵入下一 step 的读取和 H2D。
3. **全 tier 复制，并用“完成驱动”的 meet-in-the-middle 路由。**每个 chunk 在 NVMe 和 PFS 各有一份，因此任一层都能读取任意 chunk。NVMe 从需求列表前端读，PFS 从后端读，每次领取 $m$ 个 chunk；某层完成一批就立即领取下一批，直到两侧相遇。若略微跨界，少量 chunk 会被重复读取，但先到者已可使该 chunk 就绪。实现是独立 C++ `io_uring` 引擎，读写使用不同 ring；论文声称该路由自然逼近聚合带宽上界，无需在线测速或调静态比例。
4. **给出需求、带宽上界、静态分配损失和端到端收益模型。**核心关系包括 $D=V\cdot\Pr[d>M]$、$T^{ret}_i\ge D_i/(b_0(i)+b_1(i))$，以及静态比例为 $g$ 时的损失 $\max(g/\phi(i),(1-g)/(1-\phi(i)))$；其中 $d$ 是 reuse distance，$M$ 是 GPU+host 容量，$\phi(i)=b_0(i)/(b_0(i)+b_1(i))$ 随 step 波动。只有回读仍占 step 时间时，降低 $T^{ret}$ 才能通过吞吐或排队队列放大成 TTFT 收益。

## 带来什么提升

1. **离线吞吐：**在单节点 4×H100、LooGLE、TP=4 下，14B/32B/70B/110B 分别为 LMCache 的 **1.17×/1.37×/1.48×/1.49×**，为 CPU-only 的 1.15×–3.22×；x32B 上 LooGLE/NarrativeQA/CUAD 分别为 1.37×/1.21×/1.05×。CUAD 请求更短、回读占比更小，收益也最小。读卷与 LMCache 相近，提升主要来自并行读取两层存储，而非少传 KV。
2. **在线 TTFT 随负载放大：**32B 的 Qwen-Coder 在低/中/高到达率下，KVStream 平均 TTFT 相对 LMCache 提升 **1.07×/1.61×/3.12×**；Chutes 为 1.23×/1.71×/2.01×。单请求回读时间只低 15%–22%，但更快排空队列使在途请求差距扩大，因此高负载带来更大的端到端收益。SWE-bench agentic trace 中，到达 CV=4 时平均 TTFT 加速 **6.07×**，即摘要中约 6.1×。
3. **SLO 与扩展性：**SWE-bench、32B、Testbed-2、1 秒 TTFT SLO 下，KVStream 达标率 **84%**，LMCache 70%、Mooncake Store 55%、CPU-only 37.7%、SGLang HiCache 的磁盘后端 34%。在 16–128 GPU 上，吞吐相对 LMCache 保持 1.23–1.36×；32 节点时 p90 step time 约 2.7s，LMCache 在 4 节点已约 4.0s。每节点独立按完成事件调度，无需跨节点协调。
4. **机制消融：**离线 M1 批读与 M2 延后写各自不到 1%，因单 NVMe 仍封顶；加入 PFS 和固定 70:30 分配（M3）达到 1.24×，完成驱动路由（M4）达到 **1.48×**。固定比例在 1TB/0.5TB 读写竞争或 PFS 外部负载变化时出现 8.0%–67.7% 的 bound gap，KVStream 在各次测量中约在 4.3% 以内。

## 备注

- **复制写不是免费：**KVStream 将每个 chunk 同时写到两层，写开销近似 doubles；论文测到每 rank 每层约 0.19GB/s 的固定写流。收益取决于“回读量 / 复制写量”。Qwen-Coder 随到达顺序从 cache-friendly 到 round-robin 变化，该比值 0.27→1.54，TTFT 加速 1.05×→3.12×；Chutes 在 80GB/rank 时大部分复用已落入内存，速度反而约 0.98×。
- **有清晰失效区：**SWE-bench 合成降低 prefix 复用率至 4% 时，KVStream 比 LMCache 慢到 0.81×。当工作集已在 host memory、复用距离短或回读很少时，第二层复制既不做有用功还增加写压力。M3 静态按 prefix 位置放置在新 suffix 上的在线实验将平均 TTFT 推到 122.1s（LMCache 31.1s），说明“复制”与“读取路由”必须成对使用，单独复制并不等于 KVStream。
- **实验边界：**测的是 H100/A100 + 本地 NVMe RAID-0 + VAST/Lustre PFS 的 HPC 环境；云对象存储、CXL、远端 NVMe 仅作外推，未实测其尾延迟和写入能力。主结果把约 320GB host memory 留给 KV；不同 host 容量、PFS 竞争和 arrival trace 会改变收益。`within 5%` 指 retrieval time 相对聚合带宽 bound，不是端到端吞吐只差 5%。SGLang HiCache 集成写是同步且不与回读重叠，明显弱于其 vLLM+LMCache 主路径。
- **与现有 KV 工作的关系：**相比 LMCache/Mooncake 的层次缓存，KVStream 改变的是命中 storage 之后的数据路径；相比 Tutti 的 GPU-centric SSD、XM-Sparse 的专用内存节点，它要求已有且独立可达的第二存储 tier，优势来自把原本闲置的 PFS 一并纳入带宽池。它与 prefix 调度、KV 压缩、P/D 解耦正交；未来工作仍需降低复制写、加入对象存储，并研究跨节点共享同一 PFS 副本的容量与写成本。
