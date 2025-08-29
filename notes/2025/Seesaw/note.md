# Seesaw: High-throughput LLM Inference via Model Re-sharding

> Qidong Su, Wei Zhao, Xin Li, Muralidhar Andoorveedu, Chenhao Jiang, Zhanda Zhu, Kevin Song, Christina Giannoula, Gennady Pekhimenko

![](fig8.png)

## Abstract

To improve the efficiency of distributed large language model (LLM)
inference, various parallelization strategies, such as tensor and pipeline
parallelism, have been proposed. However, the distinct computational
characteristics inherent in the two stages of LLM inference-prefilling and
decoding-render a single static parallelization strategy insufficient for the
effective optimization of both stages. In this work, we present Seesaw, an LLM
inference engine optimized for throughput-oriented tasks. The key idea behind
Seesaw is dynamic model re-sharding, a technique that facilitates the dynamic
reconfiguration of parallelization strategies across stages, thereby maximizing
throughput at both phases. To mitigate re-sharding overhead and optimize
computational efficiency, we employ tiered KV cache buffering and
transition-minimizing scheduling. These approaches work synergistically to
reduce the overhead caused by frequent stage transitions while ensuring maximum
batching efficiency. Our evaluation demonstrates that Seesaw achieves a
throughput increase of up to 1.78x (1.36x on average) compared to vLLM, the
most widely used state-of-the-art LLM inference engine.
