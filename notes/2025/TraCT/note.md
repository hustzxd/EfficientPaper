# TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale

> Dongha Yoon (Virginia Tech), Younghoon Min, Hoshik Kim, Jongryool Kim (SK Hynix America), Sam H. Noh (Virginia Tech)

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Disaggregated LLM serving 通过将计算密集的 prefill 与延迟关键的 decode 分离来提升资源效率，但引入了一个根本瓶颈：prefill 产生的 KV 张量必须传输给 decode worker，现有系统依赖 RDMA 网络路径完成这一交换。本文提出 TraCT，一个 rack 规模 LLM 服务系统，将 CXL 共享内存同时用作 (1) 无网络 KV 传输底座和 (2) rack 级 prefix 感知 KV cache：GPU 通过 CXL load/store 与 DMA 直接读写 KV block，彻底消除 NIC hop。为在无跨节点原子操作、无全设备一致性的非一致性 CXL Type-3 内存上实现该设计，TraCT 提出两层节点间锁、软件管理的数据可见性、共享分配器与对象store等软件方案。基于 Dynamo 框架的实现表明：相比 RDMA 与 DRAM 缓存基线，平均 TTFT 降低至多 9.8×，P99 延迟降低至多 6.2×，峰值吞吐提升至多 1.6×。

## 一句话总结

TraCT 用 CXL Type-3 共享内存同时替代 RDMA KV 传输路径与 rack 级 prefix cache，GPU 经 GPU–CXL DMA 直接读写 KV block，在真实 CXL 硬件上把解耦式 LLM 服务的 KV 交换从网络栈搬到内存语义通路。

## 创新点

1. **KV Transfer 与 prefix Caching 紧耦合（TraCT 命名由来）**：同一 CXL 共享内存区域既当传输底座又当 rack 级 KV cache——cache-hit 的 KV block 无需任何传输（decode worker 直接 CXL→GPU DMA 读取），cache-miss 由 prefill worker GPU→CXL DMA 写入并发布，消除了 LMCache 等方案"命中也要走网络传一遍"的冗余路径。
2. **两层节点间锁（two-tier inter-node lock）**：CXL Type-3 无跨节点原子操作、无全局一致性。TraCT 用每节点 DRAM 内 local lock（pthread_mutex 数组）收敛节点内竞争，使每个全局锁最多只有"每节点一个"竞争者；全局锁阵列驻留 CXL 共享内存，由专用 lock manager 线程扫描 slot（I/W/L 三态）授权。无需硬件原子指令、无需集中式 metadata server。
3. **非一致性共享内存上的可见性纪律**：metadata（prefix 索引、引用计数、分配器状态）放紧凑的 cacheline 对齐控制区，细粒度 `clflush`（而非异步的 `clflushopt`——后者 flush 可能滞留 store buffer，mfence 也无法保证到达设备，会导致他节点读到旧值）；大块 KV payload 只经 GPU-CXL DMA、从不进 CPU cache，因此"metadata READY 发布"即可作为全局可见性边界，免去 flush 数十 MB payload。
4. **无共享指针的共享数据结构**：跨节点虚拟地址不同，所有共享结构用 64-bit 偏移寻址（ptr = base + off）；全局 chunk 分配器（CXL 内 bitmap）+ 每节点 local heap（DRAM free-list）两层分配，把分配元数据竞争从节点间收缩到节点内；对象 store 只发布少量根对象（如 prefix 哈希表），内部结构用偏移链接，优于 cMPI Arena 逐元素注册的 flat key-value 设计。
5. **静态 prefix 索引规避结构更新**：放弃 prefix tree（插入/分裂/合并都会触发锁+flush），改用固定大小线性探测哈希表 + vLLM 的迭代式 block hash（$h_i = \text{hash}(h_{i-1}, T_i)$，天然保持前缀关系），LRU 逐出仅触碰紧凑元数据字段。

## 带来什么提升

1. **纯传输路径（禁用缓存）即胜过 RDMA**：对比 Dynamo NIXL/UCX，1500–6000 token 输入的 TTFT CDF 全面左移，长 prompt（6000 token）优势最大、尾部更短；吞吐持平——证明 CXL DMA 绕过 NIC 队列、host DRAM 拷贝与传输层开销，CXL 是可行的 KV 传输底座。
2. **启用缓存后**：峰值吞吐较 LMCache（DRAM 缓存基线，48GB）提升至多 1.6×（QPS=3.0），平均 TTFT 降低至多 9.83×，P99 TTFT 降低至多 6.2×——且 TraCT 的 prefix 命中率与 LMCache 相当甚至更低，增益来自"命中零传输"而非更高命中率。
3. **延迟更稳定**：TTFT CDF 更陡峭，消除网络队列带来的方差，突发/高并发下延迟可预测性显著提升。
4. **资源与能效**：prefill/decode 的 GPU SM 占用下降（命中跳过重算、decode 不再等 host-host 传输），decode 侧 GPU RX 有效带宽更高，GPU 功耗降低，指向更低 TCO。

## 备注

- 实验规模有限：2 台服务器（各 1× A6000 48GB + 512GB DRAM），CXL 设备为 Niagara 2.0（640ns 延迟 / 10.1 GB/s，配 64GB 共享区），模型为 DeepSeek-R1-Distill-Llama-8B；CXL 带宽成为潜在规模瓶颈，仅适用于 rack 内场景。
- 与 Beluga（集中式 metadata server 方案）、cMPI（O(N²) 队列）的同步设计形成对比：TraCT 强调去中心化、纯 load/store 的节点间协调；作者称是首个在真实非一致性 CXL Type-3 硬件上支撑 rack 级 KV 缓存+传输的系统。
- 代码约 5K 行 C/C++（共享内存库 + KV connector）+ 少量 Python 包装，"计划发表后开源"，当前无公开仓库。
