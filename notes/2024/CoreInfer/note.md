# CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Large language models (LLMs) with billions of parameters have sparked a new
wave of exciting AI applications. However, their high computational costs and
memory demands during inference pose significant challenges. Adaptive sparse
activation inference, which activates only a small number of neurons for each
token, offers a novel way to accelerate model inference without degrading
performance, showing great potential for resource-constrained hardware devices.
Nevertheless, existing methods predict activated neurons based on individual
tokens with additional MLP, which involve frequent changes in activation maps
and resource calls, limiting the acceleration benefits of sparse activation. In
this paper, we introduce CoreInfer, an MLP-free adaptive sparse activation
inference method based on sentence-level prediction. Specifically, we propose
the concept of sentence-wise core neurons, which refers to the subset of
neurons most critical for a given sentence, and empirically demonstrate its
effectiveness. To determine the core neurons, we explore the correlation
between core neurons and the sentence's semantics. Remarkably, we discovered
that core neurons exhibit both stability and similarity in relation to the
sentence's semantics -- an insight overlooked by previous studies. Building on
this finding, we further design two semantic-based methods for predicting core
neurons to fit different input scenarios. In CoreInfer, the core neurons are
determined during the pre-filling stage and fixed during the encoding stage,
enabling zero-cost sparse inference. We evaluated the model generalization and
task generalization of CoreInfer across various models and tasks. Notably, on
an NVIDIA TITAN XP GPU, CoreInfer achieved a 10.33 times and 2.72 times speedup
compared to the Huggingface implementation and PowerInfer, respectively.
