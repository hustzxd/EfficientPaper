# PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design

> Xingyu Qu, Tianhao Lin, Yiqi Li, Zhiyu Chen, Sheng Wang

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Modern online large language model (LLM) services, such as Retrieval-Augmented Generation (RAG) and agent systems, increasingly expose two prominent characteristics: prompt segmentation (e.g., system instructions, retrieved passages, tool outputs) and hotspot skew, where a small set of these segments recurs frequently across user requests. Failing to jointly exploit these patterns could lead to repeated prefill of hot segments and prolonged TTFT, undermining both throughput and user-perceived responsiveness. However, existing work tackles these patterns independently: KV-cache management mainly exploits segment reuse while scheduling reorders requests to improve cache locality, yet neither aligns request admission with KV-cache retention. To address this gap, we first analyze how scheduling and KV-cache management jointly affect TTFT. Guided by this, we present PRISM (Prefix Reuse Optimization Integrated Scheduling and Memory), which co-designs a query-aware scheduler (QAS) with a demand-aware radix tree (DART) to align request admission with exact-prefix KV retention. Our evaluation results show that, versus the strongest baseline, PRISM reduces average per-QPS P99 TTFT by 23.3% and 37.1% while increasing exact-prefix KV-cache hit rate by 5.9 and 12.2 percentage points on 4B and 13B models, respectively.

## 一句话总结

PRISM 把“哪些请求应该先进入 batch”和“哪些 radix-tree KV prefix 应该在显存压力下保留”做成同一个控制闭环，用调度侧的近未来 segment demand 去指导 KV-cache retention，从而降低 RAG/Agent 在线服务的 tail TTFT。

## 创新点

1. 提出 scheduling-memory co-design 视角：论文把 TTFT 拆成 admission wait、missing-prefix prefill 和 first-token latency，并指出只做 prefix-aware scheduling 或只做 cache eviction 都可能失效，因为热点请求被排近了不代表对应 KV anchor 还在显存里。
2. QAS 用 reusable segment 的 queued、active、next-batch 三类计数生成 order-sensitive bucket signature，把共享热点 segment 的请求排进 hot lane，同时保留少量 FIFO cold lane 避免稀有 prefix 饥饿。
3. DART 在 radix KV cache 上增加 segment anchor 和 demand metadata，用 dispatch batch 的 QAS priority 保护高价值 reusable anchors，优先淘汰 private suffix 和低需求 reusable node，仍然只复用 exact token-prefix KV，避免语义近似带来的错误复用。

## 带来什么提升

1. 在 MultiHopRAG 风格主实验中，相比 strongest baseline，PRISM 让平均 per-QPS P99 TTFT 在 Qwen3-4B 上下降 23.3%，在 Llama2-13B 上下降 37.1%；相对 native SGLang 分别下降 63.5% 和 80.8%。
2. exact-prefix KV-cache hit rate 明显更高：Qwen3-4B 约 49.1% vs 43.2%，Llama2-13B 约 39.5% vs 27.2%，说明收益主要来自保住即将复用的 prefix，而不是牺牲生成质量；F1 在 sweep 中基本稳定。
3. 在 tau-bench 构造的 AGENTPREFIX workload 上也有效，P99 TTFT 相比最强 baseline 在 30/40/50 QPS 下降 24.9%/11.9%/8.5%，说明同一机制可覆盖 tool schema、policy、database record、deterministic tool observation 等 agent prompt 重复区域。

## 备注

- 该方法依赖请求侧能暴露 reusable segment identity；实际系统需要把 RAG 文档、工具 schema、policy 等 prompt 片段稳定序列化，才能让 exact-prefix hit 发生。
- 实验主要是单 A800 GPU 和固定 workload/hardware setting，多 GPU KV placement、异构 GPU pool、跨机 interconnect 影响仍是未来工作。
