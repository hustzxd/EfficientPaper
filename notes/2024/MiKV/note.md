# No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization

> June Yong Yang, Byeongwook Kim, Jeongin Bae, Beomseok Kwon, Gunho Park, Eunho Yang, Se Jung Kwon, Dongsoo Lee

![111](../../blank.jpg)

## Abstract

Key-Value (KV) Caching has become an essential technique for accelerating the
inference speed and throughput of generative Large Language Models~(LLMs).
However, the memory footprint of the KV cache poses a critical bottleneck in
LLM deployment as the cache size grows with batch size and sequence length,
often surpassing even the size of the model itself. Although recent methods
were proposed to select and evict unimportant KV pairs from the cache to reduce
memory consumption, the potential ramifications of eviction on the generative
process are yet to be thoroughly examined. In this paper, we examine the
detrimental impact of cache eviction and observe that unforeseen risks arise as
the information contained in the KV pairs is exhaustively discarded, resulting
in safety breaches, hallucinations, and context loss. Surprisingly, we find
that preserving even a small amount of information contained in the evicted KV
pairs via reduced precision quantization substantially recovers the incurred
degradation. On the other hand, we observe that the important KV pairs must be
kept at a relatively higher precision to safeguard the generation quality.
Motivated by these observations, we propose \textit{Mixed-precision KV
cache}~(MiKV), a reliable cache compression method that simultaneously
preserves the context details by retaining the evicted KV pairs in
low-precision and ensure generation quality by keeping the important KV pairs
in high-precision. Experiments on diverse benchmarks and LLM backbones show
that our proposed method offers a state-of-the-art trade-off between
compression ratio and performance, compared to other baselines.
