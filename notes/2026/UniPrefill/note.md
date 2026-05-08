# UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification

> Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He

![111](cover.jpg)

## Abstract

As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been proposed, effectively alleviating the computational burden of long-context inference. However, existing research on long-context prefill acceleration remains predominantly focused on sparse attention mechanisms, which achieve their maximum speedup only on full-attention models. When transferred to emerging architectures--such as linear/full attention hybrids or sliding window/full attention hybrids--these prefill acceleration approaches suffer significant performance degradation. Furthermore, such methods are generally incompatible with continuous batching, making them difficult to integrate into modern inference engines such as vLLM. To this end, we propose UniPrefill, a prefill acceleration framework applicable to virtually any model architecture, which directly accelerates the model's computation at the token level. We further implement UniPrefill as a continuous batching operator and extend vLLM's scheduling strategy to natively support prefill-decode co-processing and tensor parallel for UniPrefill, enabling its seamless integration into vLLM. UniPrefill achieves up to 2.1x speedup in Time-To-First-Token (TTFT), with the acceleration becoming increasingly pronounced as the number of concurrent requests grows.

用last n query预测重要性，后续层都沿用这个重要性，属于sparse prefill attention

---

*以下总结由 MiMo 生成：*

这篇论文针对长上下文大语言模型推理效率低的问题，提出了一种通用的预填充加速框架UniPrefill。该方法通过块级动态稀疏化技术，在令牌级别直接加速模型计算，适用于多种模型架构（如线性/全注意力混合架构）。UniPrefill被实现为连续批处理算子，并扩展了vLLM的调度策略以支持预填充-解码协同处理和张量并行，实现了与现代推理引擎的无缝集成。实验表明，UniPrefill在首令牌时间（TTFT）上实现了最高2.1倍的加速，且随着并发请求数量的增加，加速效果更加显著。

---

## 论文详细总结

### 1. 研究背景与动机

随着 LLM 对上下文长度需求增长，现有预填充加速方法主要依赖稀疏注意力机制，但存在以下问题：
- 仅在全注意力模型上达到最佳加速效果
- 迁移至混合架构（线性/全注意力、滑动窗口/全注意力）时性能显著下降
- 与连续批处理不兼容，难以集成到 vLLM 等现代推理引擎

### 2. UniPrefill 核心思想

**通用预填充加速框架**，适用于几乎所有模型架构，以 token 级计算为核心，实现为连续批处理算子。

### 3. 关键技术

| 技术 | 说明 |
|------|------|
| **Block-wise Dynamic Sparsification** | 块级动态稀疏化，核心加速策略 |
| **调度策略扩展** | 扩展 vLLM 调度策略，支持预填充-解码协同处理 |
| **张量并行支持** | 为 UniPrefill 扩展 tensor parallel 功能 |

### 4. 实验结果

| 指标 | 结果 |
|------|------|
| TTFT 加速 | 最高 **2.1 倍** |
| 并发效果 | 随并发请求数增加，加速更显著 |

### 5. 核心贡献

1. 提出适用于**几乎所有模型架构**的通用预填充加速框架
2. 以 token 级计算为核心，克服对特定架构的依赖
3. 实现为连续批处理算子，解决与现代推理引擎的兼容性问题
4. 扩展 vLLM 调度策略，支持预填充-解码协同处理和张量并行

> 代码：https://github.com/qhfan/UniPrefill.git
