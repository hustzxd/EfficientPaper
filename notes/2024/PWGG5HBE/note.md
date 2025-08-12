# A Survey on Large Language Model Acceleration based on KV Cache Management

![](../../blank.jpg)

## Abstract

Large Language Models (LLMs) have revolutionized a wide range of domains such
as natural language processing, computer vision, and multi-modal tasks due to
their ability to comprehend context and perform logical reasoning. However, the
computational and memory demands of LLMs, particularly during inference, pose
significant challenges when scaling them to real-world, long-context, and
real-time applications. Key-Value (KV) cache management has emerged as a
critical optimization technique for accelerating LLM inference by reducing
redundant computations and improving memory utilization. This survey provides a
comprehensive overview of KV cache management strategies for LLM acceleration,
categorizing them into token-level, model-level, and system-level
optimizations. Token-level strategies include KV cache selection, budget
allocation, merging, quantization, and low-rank decomposition, while
model-level optimizations focus on architectural innovations and attention
mechanisms to enhance KV reuse. System-level approaches address memory
management, scheduling, and hardware-aware designs to improve efficiency across
diverse computing environments. Additionally, the survey provides an overview
of both text and multimodal datasets and benchmarks used to evaluate these
strategies. By presenting detailed taxonomies and comparative analyses, this
work aims to offer useful insights for researchers and practitioners to support
the development of efficient and scalable KV cache management techniques,
contributing to the practical deployment of LLMs in real-world applications.
The curated paper list for KV cache management is in:
\href{https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management}{https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management}.
