# Týr-the-Pruner: Unlocking Accurate 50% Structural Pruning for LLMs via Global Sparsity Distribution Optimization

> Guanchen Li, Yixing Xu, Zeping Li, Ji Liu, Xuanwu Yin, Dong Li, Emad Barsoum

![111](fig2.png)

## Abstract

Structural pruning enhances hardware-agnostic inference efficiency for large
language models (LLMs) but often struggles to maintain performance. Local
pruning performs efficient layer-by-layer compression but ignores global
topology. Global pruning has the potential to find the optimal solution
although resource-intensive. However, existing methods tend to rank structural
saliency uniformly, ignoring inter-structure dependencies and failing to
achieve end-to-end optimization. To address these limitations, we propose
T\'yr-the-Pruner, an efficient end-to-end search-based global structural
pruning framework. This framework constructs a supernet by repeatedly applying
local pruning across a range of sparsity ratios to each layer in an LLM, with
the core goal of determining the optimal sparsity distribution under a target
overall sparsity ratio. Concretely, we introduce an effective local pruning and
an expectation error accumulation approach to improve supernet construction.
Furthermore, we employ an iterative prune-and-search strategy with
coarse-to-fine sparsity granularity to ensure efficient search convergence.
Experimental results show that T\'yr-the-Pruner achieves state-of-the-art
structural pruning, retaining 97% of the dense model's performance while
removing a challenging 50% of Llama-3.1-70B's parameters.

evolutionary search得到每一层的稀疏度