# Beyond Speedup -- Utilizing KV Cache for Sampling and Reasoning

> Zeyu Xing, Xing Li, Hui-Ling Zhen, Mingxuan Yuan, Sinno Jialin Pan

![111](../../blank.jpg)

## Abstract

KV caches, typically used only to speed up autoregressive decoding, encode contextual information that can be reused for downstream tasks at no extra cost. We propose treating the KV cache as a lightweight representation, eliminating the need to recompute or store full hidden states. Despite being weaker than dedicated embeddings, KV-derived representations are shown to be sufficient for two key applications: \textbf{(i) Chain-of-Embedding}, where they achieve competitive or superior performance on Llama-3.1-8B-Instruct and Qwen2-7B-Instruct; and \textbf{(ii) Fast/Slow Thinking Switching}, where they enable adaptive reasoning on Qwen3-8B and DeepSeek-R1-Distil-Qwen-14B, reducing token generation by up to $5.7\times$ with minimal accuracy loss. Our findings establish KV caches as a free, effective substrate for sampling and reasoning, opening new directions for representation reuse in LLM inference. Code: https://github.com/cmd2001/ICLR2026_KV-Embedding.
