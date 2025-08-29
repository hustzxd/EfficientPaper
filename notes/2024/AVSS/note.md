# AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis

![](avss.png)

## Abstract

The evaluation of layer importance in deep learning has been an active area
of research, with significant implications for model optimization and
interpretability. Recently, large language models (LLMs) have gained prominence
across various domains, yet limited studies have explored the functional
importance and performance contributions of individual layers within LLMs,
especially from the perspective of activation distribution. In this work, we
propose the Activation Variance-Sparsity Score (AVSS), a novel metric combining
normalized activation variance and sparsity to assess each layer's contribution
to model performance. By identifying and removing approximately the lowest 25%
of layers based on AVSS, we achieve over 90% of original model performance
across tasks such as question answering, language modeling, and sentiment
classification, indicating that these layers may be non-essential. Our approach
provides a systematic method for identifying less critical layers, contributing
to efficient large language model architectures.

根据layer activation的相似性，进行 layer pruning.