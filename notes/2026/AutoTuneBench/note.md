# AutoTuneBench: Trustworthy Measurement for Agent Auto-Tuning of LLM Serving Engines

> Li Chen

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> 来源：[arXiv:2609.18123v1](https://arxiv.org/abs/2609.18123v1)，2026-09-16，harnets.ai；[代码与测量记录 v1.0.0](https://github.com/li-ch/autotunebench/tree/v1.0.0)。已阅读正文，并静态核对发布版 README 和 `config/protocol.yaml`；未复现 GPU 实验。

## Abstract

Large language model agents tune GPU kernels and serving engines through a closed loop of propose, measure, and keep, but the measurements behind this loop are not trustworthy. We characterize four failure modes from a four-day pilot corpus of 619 model calls: strawman baselines manufacture speedups, absolute times do not transfer across machines, saturated tasks nullify comparisons, and infrastructure defects impersonate science. We present AutoTuneBench, a benchmark and measurement protocol that makes trust architectural. The protocol is frozen as code with test-enforced provenance; a database-level validator rejects out-of-protocol results; anti-cheat checks run outside the agent's modification surface; comparisons follow pre-registered readouts; and measurements anchor to externally published results, grounded in paired-seed statistics with a 5% cross-run coefficient-of-variation cap. Honest measurement rewrites the headlines: our best kernel reads 10.6x against a naive baseline but 2.03x against the honest one; one configuration delivers 1.174x on one machine and 1.0049x on another; a pre-registered on/off comparison nulls at a shared wall (2.4840 vs 2.4957 ms); and the KernelBench Level-1 suite admits 51% of tasks with median speedup 1.0001x over PyTorch eager. The protocol, the two-engine corpus (vLLM and SGLang), and its audit trail are released as open artifacts.

## 一句话总结

AutoTuneBench 将 Agent 自动调优的评估器、参考实现、实验协议和证据入库路径置于候选代码的修改范围之外，用可审计测量区分真实优化、基线选择、硬件差异、任务饱和与基础设施故障。

## 创新点

1. **把测量协议固化为代码与入库约束。** `protocol.yaml` 固定 correctness tolerance、至少 5 个输入 shape、计时设置、engine 配对 seeds 101/202、跨运行变异系数上限 5% 和费用预算；字段来源由回归测试追踪，数据库 ingest validator 拒绝可在入库时检测出的违规结果。candidate、逐轮 JSONL 和逐次模型调用 ledger 共同记录成功、拒绝、成本与失败原因。
2. **把反作弊放在 Agent 无权修改的位置。** 每个任务只开放白名单文件，参考实现和评测代码只读；结合静态扫描、被替代算子的运行时 fallback 计数、断网 sandbox、留出输入/设备复测和超过 10x 的异常隔离。10x 阈值只是人工复核触发器，不能证明低于阈值的候选正确。
3. **让实验先证明干预送达，再解释收益。** A/B 预先规定读数、有效迭代数和停止规则；首个 kept candidate 没有 profiler feedback 时立即停止。跨机器仅比较各自本地 baseline 的比值，并用公开 Astra kernel 结果校验 harness；复现偏差作为机制分析结果记录，不能为了过关事后修改阈值。

## 带来什么提升

1. **测量纠偏：**同一 GELU 优化以 naive 实现为基准得到 10.6x，以搜索实际起点为基准仅 2.03x。KernelBench Level-1 的 100 个任务中，原始 53 个正确候选有 2 个利用全正输入把 ReLU 退化为 identity，被异常检查隔离后剩 **51/100**；这 51 个任务相对 PyTorch eager 的中位加速仅 **1.0001x**，其中 24 个更慢、12 个达到至少 1.2x。53 个候选复测的 correctness verdict 全部一致，运行间偏差中位数 0.60%、最大 3.97%（§6.3，表 4）。
2. **Serving 配置搜索收益取决于引擎和基准：**GB10 上 SGLang 使用 Qwen2.5-7B-Instruct、16 requests、256-token prompts、最多 48 个输出 token，确定性 flag-space 搜索通过 ngram speculation 将 requests/s 从 0.2644 提到 0.3372，即 **1.2753x**；该扩展运行 20 次有效迭代、3.07 GPU-hours、模型 API 花费为 0。vLLM 扩展仅 **1.0091x throughput**，且基准是首个采样的 ngram 配置，不能解释为相对 stock default 的提升。这两组 engine 搜索使用确定性 proposer，收益不能归因于 LLM 搜索能力（§6.5，表 6）。
3. **揭示测量条件与搜索效率的差别：**Astra warm-cache 协议下部分 kernel 的收益在每次刷新 L2 后收敛到约 0.93–1.07x；同一饱和任务上，LLM 无/有 profiler 两组分别有 11/20、14/20 次有效迭代，指定随机变换菜单为 3/20，但最终最好时间都约 2.47–2.50 ms。论文还通过 ledger 找到 4096-token 截断、profiler 干预未送达和 patch chaining 失败，避免把这些现象误认为模型能力或算法结论（§6.1–6.2、6.4、6.6）。

## 备注

- **A/B 结论存在内部张力。**§4.3 规定有效迭代不足时实验无效且优先于其他 verdict；表 3 两组均未达到预注册的 15 次门槛，§6.2 仍称 shared-wall null。因此只能记录“已观测候选接近饱和且未见 profiler 优势”，不能据此确认 profiler 无效，也不能将单个受限随机菜单推广为随机搜索的一般上限。
- **跨设备与基线需分开解释。**主要 GPU 结果来自单台 DGX Spark GB10；M5 Max/MLX 复测是算法移植并重建本地 baseline，非同一 Triton binary 跨硬件执行。§6.7 又引用 GELU 的 11.03x 认证记录，与前文 10x 异常门槛及 2.03x honest headline 的基准口径需进一步对账；该数不能与 Mac 的约 0.79x 直接比较以量化硬件差距。冷/热 cache、算子输出语义、输入分布和 stock kernel 版本也会改变加速比。
- **可借鉴的是评估纪律。**做 kernel 生成或 cost-aware KV policy 时，应冻结独立评估器、明确 stock/首个候选/当前最优三种基准，确认策略实际生效，按冷/热与设备条件配对复测，并公开失败次数、GPU 时间和 API 成本。本论文并未验证多硬件 fleet 上的联合调优，也未证明审计机制能消除所有作弊。元数据不把 vLLM/SGLang 这两个执行引擎当成调优算法 baseline。
