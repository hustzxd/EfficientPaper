# A Simple Linear Patch Revives Layer-Pruned Large Language Models

<p align="center">
<img src="fig3.png" width="600" title="blank">
</p>

## Abstract

Layer pruning has become a popular technique for compressing large language
models (LLMs) due to its simplicity. However, existing layer pruning methods
often suffer from significant performance drops. We identify that this
degradation stems from the mismatch of activation magnitudes across layers and
tokens at the pruning interface. To address this, we propose LinearPatch, a
simple plug-and-play technique to revive the layer-pruned LLMs. The proposed
method adopts Hadamard transformation to suppress massive outliers in
particular tokens, and channel-wise scaling to align the activation magnitudes.
These operations can be fused into a single matrix, which functions as a patch
to bridge the pruning interface with negligible inference overhead. LinearPatch
retains up to 94.15% performance of the original model when pruning 5 layers of
LLaMA-3-8B on the question answering benchmark, surpassing existing
state-of-the-art methods by 4%. In addition, the patch matrix can be further
optimized with memory efficient offline knowledge distillation. With only 5K
samples, the retained performance of LinearPatch can be further boosted to
95.16% within 30 minutes on a single computing card.
