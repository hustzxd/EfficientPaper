# Online Scheduling for LLM Inference with KV Cache Constraints

> Patrick Jaillet, Jiashuo Jiang, Konstantina Mellou, Marco Molinaro, Chara Podimata, Zijie Zhou

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

论文首次从在线优化视角系统建模带 KV Cache 内存约束的 LLM 推理批处理与调度问题：请求持续到达、输出长度未知且 KV Cache 随生成 token 增长。作者构造具有未来信息的 hindsight-optimal 整数规划基准，提出 Memory-Constrained Shortest-First（MC-SF）在线算法，并同时给出不可行性边界、竞争比分析与合成/真实数据实验。

## 一句话总结

MC-SF 通过优先完成中的短请求、预测未来 KV 增长并在内存约束下尽可能填满 batch，把 LLM serving 调度从经验规则提升为可分析的在线优化问题。

## 创新点

1. **形式化 KV-aware 在线调度模型**：同时刻画非抢占执行、逐 token 生成、batch 选择、请求到达、输出长度和 KV Cache 线性增长，避免把 LLM 调度简单等同于经典作业调度。
2. **hindsight-optimal 整数规划基准**：假设已知所有未来请求及输出长度，求解最小总端到端延迟，为在线算法提供统一的性能上界/对照标准。
3. **MC-SF 算法与理论边界**：优先调度已部分完成的请求，再选择等待请求扩充 batch；通过预测执行期间的未来 KV 占用保证 batch 全程可行。论文证明任意到达过程下不存在具有常数竞争比的确定性在线算法，同时在受限到达结构和输出长度预测条件下证明 MC-SF 具有常数竞争比。

## 带来什么提升

1. **接近全知最优**：在 200 个同时到达的合成实例中，MC-SF 平均延迟比为 1.005，并有 114 个实例达到精确最优；随机在线到达场景平均比为 1.047。
2. **更好的 batch 利用率与延迟**：相比参数化 benchmark，MC-SF 在高、低请求负载的真实对话数据模拟中均显著降低总延迟；核心收益来自短请求优先和 KV 内存可行性预测的联合作用。
3. **具备部署与成本意义**：在 Llama2-70B/A100 模拟及约 21 万独立 IP 的公开对话数据上验证，调度改进可转化为更低 GPU 资源、能源消耗和推理成本。

## 备注

- 理论保证依赖相对可靠的输出长度预测；预测误差会直接影响未来 KV 占用估计和可行性判断。
- 论文重点是调度策略的理论建模与仿真验证，实际生产系统还需处理 tensor/pipeline parallelism、抢占、分页 KV 管理和多模型路由等工程因素。
- 对 EfficientPaper 的价值：为 `kv_cache_management`、`performance_modeling` 与 `deployment` 三个方向提供可证明的调度基线，可与 Sorted-F 等变量 prefill/decode 长度优化工作形成理论互补。
