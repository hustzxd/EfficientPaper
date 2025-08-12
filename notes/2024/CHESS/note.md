# CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification

![](../../blank.jpg)

## Abstract

Deploying large language models (LLMs) on edge devices presents significant
challenges due to the substantial computational overhead and memory
requirements. Activation sparsification can mitigate these challenges by
reducing the number of activated neurons during inference. Existing methods
typically employ thresholding-based sparsification based on the statistics of
activation tensors. However, these methods do not explicitly model the impact
of activation sparsification on performance, leading to suboptimal performance
degradation. To address this issue, this paper reformulates the activation
sparsification problem by introducing a new objective that optimizes the
sparsification decisions. Building on this reformulation, we propose CHESS, a
general activation sparsification approach via CHannel-wise thrEsholding and
Selective Sparsification. First, channel-wise thresholding assigns a unique
threshold to each activation channel in the feed-forward network (FFN) layers.
Then, selective sparsification involves applying thresholding-based activation
sparsification to specific layers within the attention modules. Finally, we
detail the implementation of sparse kernels to accelerate LLM inference.
Experimental results demonstrate that the proposed CHESS achieves lower
performance degradation over 8 downstream tasks while activating fewer
parameters compared to existing methods, thus speeding up the LLM inference by
up to 1.27x.
