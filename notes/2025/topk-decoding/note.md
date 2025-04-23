# Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

<p align="center">
<img src="alg1.png" width="600" title="blank">
</p>

## Abstract

There is growing demand for performing inference with hundreds of thousands
of input tokens on trained transformer models. Inference at this extreme scale
demands significant computational resources, hindering the application of
transformers at long contexts on commodity (i.e not data center scale)
hardware. To address the inference time costs associated with running
self-attention based transformer language models on long contexts and enable
their adoption on widely available hardware, we propose a tunable mechanism
that reduces the cost of the forward pass by attending to only the most
relevant tokens at every generation step using a top-k selection mechanism. We
showcase the efficiency gains afforded by our method by performing inference on
context windows up to 1M tokens using approximately 16GB of GPU RAM. Our
experiments reveal that models are capable of handling the sparsity induced by
the reduced number of keys and values. By attending to less than 2% of input
tokens, we achieve over 95% of model performance on common benchmarks (RULER,
AlpacaEval, and Open LLM Leaderboard).

在CPU上计算qk，得到attention score，选取topk搬到gpu上计算