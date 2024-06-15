# Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Exploiting activation sparsity is a promising approach to significantly
accelerating the inference process of large language models (LLMs) without
compromising performance. However, activation sparsity is determined by
activation functions, and commonly used ones like SwiGLU and GeGLU exhibit
limited sparsity. Simply replacing these functions with ReLU fails to achieve
sufficient sparsity. Moreover, inadequate training data can further increase
the risk of performance degradation. To address these challenges, we propose a
novel dReLU function, which is designed to improve LLM activation sparsity,
along with a high-quality training data mixture ratio to facilitate effective
sparsification. Additionally, we leverage sparse activation patterns within the
Feed-Forward Network (FFN) experts of Mixture-of-Experts (MoE) models to
further boost efficiency. By applying our neuron sparsification method to the
Mistral and Mixtral models, only 2.5 billion and 4.3 billion parameters are
activated per inference iteration, respectively, while achieving even more
powerful model performance. Evaluation results demonstrate that this sparsity
achieves a 2-5x decoding speedup. Remarkably, on mobile phones, our
TurboSparse-Mixtral-47B achieves an inference speed of 11 tokens per second.
Our models are available at \url{https://huggingface.co/PowerInfer}
