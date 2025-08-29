# La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation

> Kai Liu, Bowen Xu, Shaoyu Wu, Xin Chen, Hao Zhou, Yongliang Tao, Lulu Hu

![](../../blank.jpg)

## Abstract

Activation sparsity can reduce the computational overhead and memory
transfers during the forward pass of Large Language Model (LLM) inference.
Existing methods face limitations, either demanding time-consuming recovery
training that hinders real-world adoption, or relying on empirical
magnitude-based pruning, which causes fluctuating sparsity and unstable
inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse
Activation), a novel method for activation sparsification designed to improve
LLM efficiency without requiring additional training or magnitude-based
pruning. We leverage layerwise orthogonal rotations to transform input
activations into rotated forms that are more suitable for sparsification. By
employing a Top-K selection approach within the rotated activations, we achieve
consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA
is effective across various sizes and types of LLMs, demonstrating minimal
performance degradation and robust inference acceleration. Specifically, for
LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a
consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in
zero-shot tasks compared to the dense model to just 0.54%, while surpassing
TEAL by 1.77% and CATS by 17.14%.
