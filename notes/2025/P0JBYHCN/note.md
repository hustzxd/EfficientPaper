# Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems

> En-Ming Huang, Li-Shang Lin, Chun-Yi Lee

![111](cover.png)

## Abstract

Large Language Models (LLMs) have achieved impressive results across various tasks, yet their high computational demands pose deployment challenges, especially on consumer-grade hardware. Mixture of Experts (MoE) models provide an efficient solution through selective activation of parameter subsets, which reduces computation requirements. Despite this efficiency, state-of-the-art MoE models still require substantial memory beyond typical consumer GPU capacities. Traditional offloading methods that transfer model weights between CPU and GPU introduce latency, limiting inference performance. This paper presents a novel CPU-GPU collaborative inference framework that incorporates an expert caching mechanism on the GPU to reduce data transfer requirements and enable faster inference through cache hits. Computations are offloaded to CPU for efficient cache miss handling, which benefits from CPU multithreading optimizations. The evaluations of our framework demonstrate performance improvements and highlight the potential of CPU-GPU collaboration to maximize hardware utilization for single-request inference scenarios on consumer-grade systems. The implementation of our framework is available at https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference.
