# Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

> Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, Se-Young Yun

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Scaling language models unlocks impressive capabilities, but the accompanying
computational and memory demands make both training and deployment expensive.
Existing efficiency efforts typically target either parameter sharing or
adaptive computation, leaving open the question of how to attain both
simultaneously. We introduce Mixture-of-Recursions (MoR), a unified framework
that combines the two axes of efficiency inside a single Recursive Transformer.
MoR reuses a shared stack of layers across recursion steps to achieve parameter
efficiency, while lightweight routers enable adaptive token-level thinking by
dynamically assigning different recursion depths to individual tokens. This
allows MoR to focus quadratic attention computation only among tokens still
active at a given recursion depth, further improving memory access efficiency
by selectively caching only their key-value pairs. Beyond these core
mechanisms, we also propose a KV sharing variant that reuses KV pairs from the
first recursion, specifically designed to decrease prefill latency and memory
footprint. Across model scales ranging from 135M to 1.7B parameters, MoR forms
a new Pareto frontier: at equal training FLOPs and smaller model sizes, it
significantly lowers validation perplexity and improves few-shot accuracy,
while delivering higher throughput compared with vanilla and existing recursive
baselines. These gains demonstrate that MoR is an effective path towards
large-model quality without incurring large-model cost.

每一层循环计算几次，再计算下一层，本质上是parameter sharing或者adaptive computation，代表是Recursive Transformer，MoR在Recursive Transformer基础上进行改进，每个Token采用不同的循环次数，从而使得Attention计算变成稀疏运算。
