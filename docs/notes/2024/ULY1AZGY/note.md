# Enabling High-Sparsity Foundational Llama Models with Efficient Pretraining and Deployment

![](../../blank.jpg)

## Abstract

Large language models (LLMs) have revolutionized Natural Language Processing
(NLP), but their size creates computational bottlenecks. We introduce a novel
approach to create accurate, sparse foundational versions of performant LLMs
that achieve full accuracy recovery for fine-tuning tasks at up to 70%
sparsity. We achieve this for the LLaMA-2 7B model by combining the SparseGPT
one-shot pruning method and sparse pretraining of those models on a subset of
the SlimPajama dataset mixed with a Python subset of The Stack dataset. We
exhibit training acceleration due to sparsity on Cerebras CS-3 chips that
closely matches theoretical scaling. In addition, we establish inference
acceleration of up to 3x on CPUs by utilizing Neural Magic's DeepSparse engine
and 1.7x on GPUs through Neural Magic's nm-vllm engine. The above gains are
realized via sparsity alone, thus enabling further gains through additional use
of quantization. Specifically, we show a total speedup on CPUs for
sparse-quantized LLaMA models of up to 8.6x. We demonstrate these results
across diverse, challenging tasks, including chat, instruction following, code
generation, arithmetic reasoning, and summarization to prove their generality.
This work paves the way for rapidly creating smaller and faster LLMs without
sacrificing accuracy.
