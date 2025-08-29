# LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference

> Qichen Fu, Minsik Cho, Thomas Merth, Sachin Mehta, Mohammad Rastegari, Mahyar Najibi

![111](fig3.png)

## Abstract

The inference of transformer-based large language models consists of two
sequential stages: 1) a prefilling stage to compute the KV cache of prompts and
generate the first token, and 2) a decoding stage to generate subsequent
tokens. For long prompts, the KV cache must be computed for all tokens during
the prefilling stage, which can significantly increase the time needed to
generate the first token. Consequently, the prefilling stage may become a
bottleneck in the generation process. An open question remains whether all
prompt tokens are essential for generating the first token. To answer this, we
introduce a novel method, LazyLLM, that selectively computes the KV for tokens
important for the next token prediction in both the prefilling and decoding
stages. Contrary to static pruning approaches that prune the prompt at once,
LazyLLM allows language models to dynamically select different subsets of
tokens from the context in different generation steps, even though they might
be pruned in previous steps. Extensive experiments on standard datasets across
various tasks demonstrate that LazyLLM is a generic method that can be
seamlessly integrated with existing language models to significantly accelerate
the generation without fine-tuning. For instance, in the multi-document
question-answering task, LazyLLM accelerates the prefilling stage of the LLama
2 7B model by 2.34x while maintaining accuracy.

- Prefill
    - 随着每层的运算，Token逐渐缩小
    - Token的重要性评估根据第$l$层的attention score估计，并对$l+1$层的Token进行pruning
    - 后续的layer选择只能时之前层的subset
- Decode
    - Decode时每层可以选择不同的subset，通过AuxCache实现（但论文说的比较模糊）

[ICLR2025 reject](https://openreview.net/forum?id=am5Z8dXoaV)

![](fig4.png)