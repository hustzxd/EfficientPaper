# Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity

<p align="center">
<img src="fig2.png" width="600" title="blank">
</p>

## Abstract

The increasing context window size in Large Language Models (LLMs), such as
the GPT and LLaMA series, has improved their ability to tackle complex,
long-text tasks, but at the cost of inference efficiency, particularly
regarding memory and computational complexity. Existing methods, including
selective token retention and window-based attention, improve efficiency but
risk discarding important tokens needed for future text generation. In this
paper, we propose an approach that enhances LLM efficiency without token loss
by reducing the memory and computational load of less important tokens, rather
than discarding them.We address two challenges: 1) investigating the
distribution of important tokens in the context, discovering recent tokens are
more important than distant tokens in context, and 2) optimizing resources for
distant tokens by sharing attention scores across layers. The experiments show
that our method saves $35\%$ KV cache without compromising the performance.

- Step1: 查找相邻层相近的attention score，把多个layer组成一个block；
- Step2: block内部共享attention socre，因此就减少了key 参与的运算，block内只保留一份key，value cache仍然全部都要保留；
- Step3: 训练5B tokens，结果提升