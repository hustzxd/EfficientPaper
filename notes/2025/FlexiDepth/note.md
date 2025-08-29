# Adaptive Layer-skipping in Pre-trained LLMs

![](fig2.png)

## Abstract

Various layer-skipping methods have been proposed to accelerate token
generation in large language models (LLMs). However, they have overlooked a
fundamental question: How do computational demands vary across the generation
of different tokens? In this work, we introduce FlexiDepth, a method that
dynamically adjusts the number of Transformer layers used in text generation.
By incorporating a plug-in router and adapter, FlexiDepth enables adaptive
layer-skipping in LLMs without modifying their original parameters. Introducing
FlexiDepth to Llama-3-8B model achieves layer skipping of 8 layers out of 32,
and meanwhile maintains the full 100\% benchmark performance. Experimental
results with FlexiDepth demonstrate that computational demands in LLMs
significantly vary based on token type. Specifically, generating repetitive
tokens or fixed phrases requires fewer layers, whereas producing tokens
involving computation or high uncertainty requires more layers. Interestingly,
this adaptive allocation pattern aligns with human intuition. To advance
research in this area, we open sourced FlexiDepth and a dataset documenting
FlexiDepth's layer allocation patterns for future exploration.

- 复杂一些的router设计
  - 相较于单个Linear Layer，效果更好
- Attention skipping
  - 跳过query对应的运算，但仍然计算KV cache
- MLP Skipping
  - 使用更小的MLP替换，而不是直接去掉