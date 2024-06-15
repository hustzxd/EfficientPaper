# PowerInfer-2: Fast Large Language Model Inference on a Smartphone

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

This paper introduces PowerInfer-2, a framework designed for high-speed
inference of Large Language Models (LLMs) on smartphones, particularly
effective for models whose sizes exceed the device's memory capacity. The key
insight of PowerInfer-2 is to utilize the heterogeneous computation, memory,
and I/O resources in smartphones by decomposing traditional matrix computations
into fine-grained neuron cluster computations. Specifically, PowerInfer-2
features a polymorphic neuron engine that adapts computational strategies for
various stages of LLM inference. Additionally, it introduces segmented neuron
caching and fine-grained neuron-cluster-level pipelining, which effectively
minimize and conceal the overhead caused by I/O operations. The implementation
and evaluation of PowerInfer-2 demonstrate its capability to support a wide
array of LLM models on two smartphones, achieving up to a 29.2x speed increase
compared with state-of-the-art frameworks. Notably, PowerInfer-2 is the first
system to serve the TurboSparse-Mixtral-47B model with a generation rate of
11.68 tokens per second on a smartphone. For models that fit entirely within
the memory, PowerInfer-2 can achieve approximately a 40% reduction in memory
usage while maintaining inference speeds comparable to llama.cpp and MLC-LLM.
For more details, including a demonstration video, please visit the project
site at www.powerinfer.ai/v2.
