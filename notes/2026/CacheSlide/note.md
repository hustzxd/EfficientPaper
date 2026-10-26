# CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving

> Yang Liu, Yunfei Gu, Liqiang Zhang, Chentao Wu, Guangtao Xue, Jie Li, Minyi Guo, Junhao Hu, Jie Meng | FAST 2026

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

[论文](https://www.usenix.org/system/files/fast26-liu-yang.pdf) | [会议页面](https://www.usenix.org/conference/fast26/presentation/liu-yang) | [代码](https://github.com/SJTU-Storage-Lab/CacheSlide)

## Abstract

Agent 的输入经常交错包含固定段与动态更新段：固定段内容及相对顺序不变，绝对位置却随更新段长度改变。CacheSlide 针对这一模式联合设计位置编码、局部 KV 修正和存储管理，以减少重复 prefill、保持输出质量，并降低容量压力下的 SSD 写回成本。

## 一句话总结

用较稳定的分段位置编码支持固定段跨位置复用，再选择性修复其对更新段的依赖，并把加载、计算与写回并行化。

## 创新点

1. **RPDC 与 CCPE**：定义固定段相对顺序保持不变的缓存模式；基于 CoPE 和任务模板建立稳定编码范围，降低位置变化带来的 KV 偏差。
2. **Weighted Correction Attention**：浅层识别偏差最大的 token，深层只重算选中部分，再按偏差融合新旧 KV；每四层用 CKSim 调整候选集合，修复固定段与更新段的交互。
3. **SLIDE 存储管理**：将重算 KV 暂写到新页，解除同层加载与写入的依赖；优先淘汰 clean 页，并按选中 token 数组织 dirty 页合并写回。

## 带来什么提升

1. **首 token 延迟**：图 10 中，相比仅复用前缀的 ContextCache，TTFT 缩短至原来的约 1/2.4–1/3.3；相比 CacheBlend 约为 1/1.21–1/2.11，并取得更好的任务质量。
2. **容量与 I/O**：图 12 中，SSD 写放大降低 3.11–3.62 倍；相比 PromptCache，GPU 存储需求降低 1.63–1.9 倍。
3. **吞吐稳定性**：图 13 的 batch=8 实验中，相比 CacheBlend/EPIC，Mistral-7B 吞吐提升 45.2%–49.6%，MPT-30B 提升 75%–82.2%，吞吐波动也减小。

## 备注

- 实现基于 vLLM 0.8.5；测试 Mistral-7B、MPT-30B、Llama-3 70B，以及 Reflexion、MemGPT、SWE-Agent。主要使用 A100 80 GB，70B 使用两张卡。
- CacheSlide 通过 LoRA 继续预训练启用 CoPE，比较基线保留原始 RoPE/ALiBi；结果同时包含编码适配与缓存策略收益。
- 参数扫描中，正确完成任务的 QPS 在 top-k 约 **26%**、CKSim 阈值约 **0.12** 时较优。0.12 不是重算比例，且最优值依赖负载。
