# AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models

> Yifeng Gu, Zicong Jiang, Jianxiu Jin, Kailing Guo, Ziyang Zhang, Xiangmin Xu

![](../../blank.jpg)

## Abstract

Large Language Models (LLMs) have significantly advanced the field of
Artificial Intelligence. However, their deployment is resource-intensive, not
only due to the large number of model parameters but also because the
(Key-Value) KV cache consumes a lot of memory during inference. While several
works propose reducing the KV cache by evicting the unnecessary tokens, these
approaches rely on accumulated attention score as eviction score to quantify
the importance of the token. We identify the accumulated attention score is
biased and it decreases with the position of the tokens in the mathematical
expectation. As a result, the retained tokens concentrate on the initial
positions, limiting model's access to global contextual information. To address
this issue, we propose Adaptive holistic attention KV (AhaKV), it addresses the
bias of the accumulated attention score by adaptively tuning the scale of
softmax according the expectation of information entropy of attention scores.
To make use of the holistic attention information in self-attention mechanism,
AhaKV utilize the information of value vectors, which is overlooked in previous
works, to refine the adaptive score. We show theoretically that our method is
well suited for bias reduction. We deployed AhaKV on different models with a
fixed cache budget. Experiments show that AhaKV successfully mitigates bias and
retains crucial tokens across global context and achieve state-of-the-art
results against other related work on several benchmark tasks.
