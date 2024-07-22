# Q-Sparse: All Large Language Models can be Fully Sparsely-Activated

<p align="center">
<img src="q-sparse.png" width="600" title="blank">
</p>

## Abstract

We introduce, Q-Sparse, a simple yet effective approach to training
sparsely-activated large language models (LLMs). Q-Sparse enables full sparsity
of activations in LLMs which can bring significant efficiency gains in
inference. This is achieved by applying top-K sparsification to the activations
and the straight-through-estimator to the training. The key results from this
work are, (1) Q-Sparse can achieve results comparable to those of baseline LLMs
while being much more efficient at inference time; (2) We present an
inference-optimal scaling law for sparsely-activated LLMs; (3) Q-Sparse is
effective in different settings, including training-from-scratch,
continue-training of off-the-shelf LLMs, and finetuning; (4) Q-Sparse works for
both full-precision and 1-bit LLMs (e.g., BitNet b1.58). Particularly, the
synergy of BitNet b1.58 and Q-Sparse (can be equipped with MoE) provides the
cornerstone and a clear path to revolutionize the efficiency, including cost
and energy consumption, of future LLMs.

Q-Sparse通过对激活进行top-K稀疏化和直通估计器的训练，实现了LLMs的完全稀疏激活，从而在推理时带来了显著的效率提升。同时，论文还提出了适用于稀疏激活LLMs的推理最优缩放定律。
没有真实的加速实现，同时也没有给实际的加速比