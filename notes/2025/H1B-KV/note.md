# H1B-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference

> Harshil Vejendla

![111](../../blank.jpg)

## Abstract

Autoregressive decoding in large language models (LLMs) requires caching a
growing list of past key-value (KV) pairs, making long-context inference a
memory-bound problem. While recent methods have explored quantizing the cache,
evicting tokens, or using binary sketches for keys (e.g., Loki), these
approaches often provide an incomplete solution by leaving one component (like
values) uncompressed or by discarding context information. This paper
introduces the Hybrid One-Bit KV Cache (H1B-KV), a comprehensive compression
scheme that radically reduces memory usage without sacrificing context. H1B-KV
represents each key vector using a 1-bit binary sketch, enabling
hardware-friendly bitwise attention, and further compresses value vectors using
4-bit quantization. This holistic, hybrid approach allows a 7-billion parameter
LLM to handle an 8k-token context with under 60 MB of cache memory - a 70x
reduction. We demonstrate that after a lightweight finetuning, H1B-KV matches
full-precision performance not only on perplexity benchmarks but also on
complex downstream tasks like mathematical reasoning (GSM8K), multi-task
understanding (MMLU), and code generation (HumanEval). Our results show H1B-KV
significantly outperforms leading quantization (KIVI), token eviction
(SparseLLM), and key-only sketching (Loki) methods in quality-per-byte,
establishing it as a robust solution for deploying LLMs in memory-constrained
environments.
