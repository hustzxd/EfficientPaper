# The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

![](tb1.png)

## Abstract

Sparse attention offers a promising strategy to extend long-context
capabilities in Transformer LLMs, yet its viability, its efficiency-accuracy
trade-offs, and systematic scaling studies remain unexplored. To address this
gap, we perform a careful comparison of training-free sparse attention methods
at varying model scales, sequence lengths, and sparsity levels on a diverse
collection of long-sequence tasks-including novel ones that rely on natural
language while remaining controllable and easy to evaluate. Based on our
experiments, we report a series of key findings: 1) an isoFLOPS analysis
reveals that for very long sequences, larger and highly sparse models are
preferable to smaller and dense ones. 2) The level of sparsity attainable while
statistically guaranteeing accuracy preservation is higher during decoding than
prefilling, and correlates with model size in the former. 3) There is no clear
strategy that performs best across tasks and phases, with different units of
sparsification or budget adaptivity needed for different scenarios. Even
moderate sparsity levels often result in significant performance degradation on
at least one task, highlighting that sparse attention is not a universal
solution. 4) We introduce and validate novel scaling laws specifically tailored
for sparse attention, providing evidence that our findings are likely to hold
true beyond our range of experiments. Through these insights, we demonstrate
that sparse attention is a key tool to enhance the capabilities of Transformer
LLMs for processing longer sequences, but requires careful evaluation of
trade-offs for performance-sensitive applications.
