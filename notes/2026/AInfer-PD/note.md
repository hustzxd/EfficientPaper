# AInfer-PD: Communication-Safe In-Place Prefill-Decode Multiplexing for Distributed MoE Rollouts

> Guowei Wang, Chaokun Yang, Zhenxuan Pan, Yuhong Guo, Minghua Zhu, Zhechuan Zhang, Shuo Wan, Xiaowei Zhu
>
> Ant Group
>
> arXiv:2609.00993v1, 2026

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

在大规模强化学习的 agentic rollout 中，多轮轨迹会不断产生新的 prefill，而其他轨迹仍处于 decode，因而 prefill/decode 共存是持续状态。P/D 解耦需要独立设备池和 KV 传输；in-place multiplexing 虽共享设备、模型权重和 KV，但分布式 MoE 中交叉的 attention TP/DP collective 可能以不一致顺序提交，并且 DeepEP 的 prefill 与 decode 路径共享可变协议状态。AInfer-PD 通过跨 rank 的 collective 顺序协调和 DeepEP P/D 通信状态隔离，使二者能够在同一设备上安全并发。单节点 prefill-intensive workload 相比关闭 multiplexing 的 AInfer，rollout completion time 降低 7.1%--22.5%；相比 SGLang 降低 24.8%--32.9%。双节点场景分别降低 18.0%--35.3% 和 18.3%--31.8%。

## 一句话总结

AInfer-PD 将分布式 MoE rollout 的 P/D multiplexing 从“可调度”推进到“通信协议安全”：用 rank-aligned turnstile 消除交叉 collective 的进度环，用 phase-owned DeepEP state 支持 normal-P 与 low-latency-D 并行，同时保留共享模型权重、KV 和 GPU 池。

## 创新点

1. **Rank-aligned segment turnstile**：把长 prefill 切成由通信安全边界分隔的 segment，每轮先在所有 rank 上以一致顺序提交一个 decode iteration，再释放下一段 prefill；只约束冲突 collective，其他 prefill 计算和非冲突通信继续异步执行。
2. **Selective device ordering for crossed ADP/ATP paths**：针对 decode 的 DP-attention ReduceScatter/AllGather 与 prefill 的全 TP AllReduce 交叉场景，在共享 stream 上建立统一的 D→P 顺序，避免不同 rank 以相反顺序驻留 kernel 而形成全局 progress cycle。
3. **DeepEP phase-owned communication state**：P 和 D 共享进程级 NVSHMEM runtime、拓扑、物理链路、权重与 KV，但分别拥有 buffer、counter、workspace、event、stream 和 QP range；P 的 normal dispatch 与 D 的 low-latency dispatch 因此可以并发。
4. **通信边界与 rollout 生命周期协同**：DeepEP P dispatch 在 expert notification 后、数据搬运前暴露安全边界；调度器同时保护 D 的 KV slot、处理 CUDA Graph replay、MTP、取消和请求生命周期，避免未完成 P 工作污染后续 KV 复用。

## 带来什么提升

1. 在单节点四类 prefill-intensive profile 上，相比同一 AInfer engine 且关闭 P/D multiplexing，固定工作量 rollout completion time 降低 **7.1%--22.5%**；相比 SGLang 降低 **24.8%--32.9%**。
2. 在双节点、16 张 H20-3E 的 EP16/ADP8 场景，E2E completion time 相比 Normal 降低 **18.0%--35.3%**，相比 SGLang 降低 **18.3%--31.8%**；live RL 的 32 个 step 汇总 rollout 时间降低 **17.6%**（17.14 h → 14.12 h）。
3. 细粒度 segment boundary 相比 whole-epoch asynchronous enqueue，E2E completion time 进一步降低 **8.6%--19.8%**；在 crossed topology 的测量中，mean D wait 从约 114--127 ms 降至约 17.5--18.0 ms。
4. DeepEP 场景中，AInfer-PD 相比 Normal 的 request-rate 提升为 **21.8%--31.9%**；phase isolation 在 EP8 中额外占用约 **465.2 MiB/rank** 的通信缓冲，P normal path 使用约 **12/132 SMs**，这是获得并发安全性的明确资源代价。

## 备注

- 主要目标是固定 rollout 工作量的 completion/makespan，而不是单请求 TTFT；激进的 P admission 会造成 TTFT 代价。单节点 profile 中 p99 TTFT 相对 Normal 为 **-6.3% 到 +1.2%**，双节点 workload 中上升 **13.2%--37.9%**。
- selective ordering 只对已注册、参与者匹配且具有安全 P boundary 的冲突 collective 给出进度论证，不等价于任意 NCCL collective 的普遍 deadlock-freedom；论文实验主要覆盖 H20-3E 的一到两节点部署。
- 对 EfficientPaper 的关系：该工作连接了 `deployment`、`overlap`、MoE expert communication 和 agentic RL workflow。它提示 P/D serving 的关键问题不仅是资源切分或 KV 搬运，还包括跨 rank 的 collective contract、phase-specific mutable state 和 rollout-level critical path。
