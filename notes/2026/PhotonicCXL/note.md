# A Photonic-CXL Memory Appliance for Scalable KV Cache Management in LLM Inference

> Jing Ding, Yash Nishant, Chandrish Ambati, Jyothsna Kamati, Trung Diep

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

LLM 推理的 KV Cache 同时需要 TB 级容量和百 GB/s 级带宽，现有 HBM、主机 DRAM、SSD 和 electrical CXL 难以兼顾。本文提出 Marvell Photonic Fabric Memory Appliance：用无源光纤 shuffle 替代 electrical CXL switch，构建无交换机的 full-crossbar 光子-CXL 混合架构，在 16 台主机之间共享 32 TB 内存，每台主机提供 128 GB/s 单向带宽。仿真和 emulation 显示，相比 electrical CXL pool 延迟降低超过 50%，多轮对话 workload 的 TTFT 最高改善 6.6 倍。

## 一句话总结

用 switch-free photonic-CXL memory pool 把 KV Cache 的容量墙和带宽墙同时后移，让长上下文、多轮对话的 cache eviction 不再直接造成 TTFT 崩溃。

## 创新点

1. **容量-带宽缺口刻画**：跨 A100/H100/H200、8B--405B 模型和最长 4M context 评估 KV retrieval，量化 host memory、SSD 和重算的速度/容量边界，指出实际 serving 需要数十 TB 容量与超过 100 GB/s 带宽。
2. **Photonic Fabric Memory Appliance**：以 16×16 passive fiber shuffle 连接 16 个 Photonic Fabric Memory Module，避免 electrical CXL switch 的 hop latency、铜缆距离限制和 retimer 功耗，提供 32 TB 共享 DDR5 memory。
3. **CXL.mem 软硬件接入**：通过 PCIe Gen6/CXL 3.1 Type-3 PF-NIC 暴露 byte-addressable memory；GPU DMA 负责 KV payload，CPU load/store 负责 metadata，并设计 KV connector 与 vLLM/SGLang 等 serving stack 对接。
4. **Serving-level validation**：将实测/emulation 得到的带宽和延迟参数接入 LLMServingSim，模拟 continuous batching、P/D disaggregation、动态 decode 和多层 memory hierarchy，而不只报告互连微基准。

## 带来什么提升

1. Host memory retrieval 相比 GPU 重算最高约 **100 倍加速**，但只能支撑几十个长上下文并发用户；SSD 容量更大却受带宽限制，最高 retrieval speedup 约 **9.7 倍**，部分场景甚至不如重算。
2. PF Memory Appliance 提供 **32 TB** 共享容量、每主机 **128 GB/s** 单向带宽并支持 **16 hosts**；emulation 相比 electrical CXL switched pool 延迟降低超过 **50%**，多种 access pattern 可达到链路满带宽利用。
3. 在 LLaMA 405B、8×H200、10 轮多轮对话、每轮 5,120 输入/500 输出 token 的仿真中，PF 使 TTFT 在 50--300 conversations 间保持约 **2,690 ms**；2 TB host DRAM baseline 在 300 conversations 时因 LRU eviction 达到约 6.6 倍更高 TTFT。
4. PF 将 prefix hit rate 在高并发下维持约 **82%**，相比 baseline 从 82% 下降到 **6.35%**；论文同时指出端到端结果主要来自 emulation 参数和 LLMServingSim，真实 PF 硬件上的完整 serving 验证尚未完成。

## 备注

- 该工作属于 deployment / hardware-system co-design，核心收益是共享 KV memory 的容量、带宽和访问延迟组合，不是新的 KV eviction 或 compression 算法。
- 论文的主要限制是 PF appliance 的端到端物理硬件仍待验证，当前 characterization 主要覆盖 8B/405B 模型，MoE 变体和成本效益比较留作后续工作。
