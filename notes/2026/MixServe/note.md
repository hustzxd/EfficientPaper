# MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

> Bowen Zhou, Jinrui Jia, Wenhao He, Yong Zhang, Fang Dong

![111](../../blank.jpg)

## Abstract

The Mixture of Experts (MoE) models are emerging as the latest paradigm for Large Language Models (LLMs). However, due to memory constraints, MoE models with billions or even trillions of parameters can only be deployed in multi-GPU or even multi-node & multi-GPU based serving systems. Thus, communication has became a major bottleneck in distributed serving systems, especially inter-node communication. Contemporary distributed MoE models are primarily implemented using all-reduce (AR) based tensor parallelism (TP) and all-to-all (A2A) based expert parallelism (EP). However, TP generally exhibits low inter-node efficiency and is thus confined to high-speed intra-node bandwidth. In contrast, EP tends to suffer from load imbalance, especially when the parallel degree is high.
  In this work, we introduce MixServe, a novel automatic distributed serving system for efficient deployment of MoE models by a novel TP-EP hybrid parallelism based on fused AR-A2A communication algorithm. MixServe begins by evaluating the communication overhead associated with various parallel strategies, taking into account the model hyperparameters and the configurations of network and hardware resources, and then automatically selects the most efficient parallel strategy. Then, we propose the TP-EP hybrid parallelism based on fused AR-A2A communication algorithm that overlaps intra-node AR communication and inter-node A2A communication. Extensive experiments on DeepSeek-R1 and Qwen3 models demonstrate that MixServe achieves superior inference performance, with 1.08~3.80x acceleration in time to first token (TTFT), 1.03~1.66x acceleration in inter-token latency (ITL), and 5.2%~50.3% throughput improvement compared to existing approaches.
