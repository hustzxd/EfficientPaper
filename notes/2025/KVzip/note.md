# KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

> Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song

![111](cover.png)

## Abstract

Transformer-based large language models (LLMs) cache context as key-value (KV) pairs during inference. As context length grows, KV cache sizes expand, leading to substantial memory overhead and increased attention latency. This paper introduces KVzip, a query-agnostic KV cache eviction method enabling effective reuse of compressed KV caches across diverse queries. KVzip quantifies the importance of a KV pair using the underlying LLM to reconstruct original contexts from cached KV pairs, subsequently evicting pairs with lower importance. Extensive empirical evaluations demonstrate that KVzip reduces KV cache size by $3$-$4\times$ and FlashAttention decoding latency by approximately $2\times$, with negligible performance loss in question-answering, retrieval, reasoning, and code comprehension tasks. Evaluations include various models such as LLaMA3.1, Qwen2.5, and Gemma3, with context lengths reaching up to 170K tokens. KVzip significantly outperforms existing query-aware KV eviction methods, which suffer from performance degradation even at a 90% cache budget ratio under multi-query scenarios.

- Step1: prefill，输入context，得到所有的kv
- Step2: 再次 Prefill, 输入 “Repeat the previous context:” context，此时不需要保留kv，也不需要保留输出，只需要把之前kv 对应的重要性scores 记录统计出来 （相当于又进行了一次prefill，prefill变慢）
- Step3: 根据重要性，把一部分kv 驱逐
- Step4: 在压缩后的kv上进行推理，速度变快