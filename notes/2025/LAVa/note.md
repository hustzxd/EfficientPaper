# LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation

> Yiqun Shen, Song Yuan, Zhengze Zhang, Xiaoliang Wang, Daxin Jiang, Nguyen Cam-Tu

![111](../../blank.jpg)

## Abstract

KV Cache is commonly used to accelerate LLM inference with long contexts, yet
its high memory demand drives the need for cache compression. Existing
compression methods, however, are largely heuristic and lack dynamic budget
allocation. To address this limitation, we introduce a unified framework for
cache compression by minimizing information loss in Transformer residual
streams. Building on it, we analyze the layer attention output loss and derive
a new metric to compare cache entries across heads, enabling layer-wise
compression with dynamic head budgets. Additionally, by contrasting cross-layer
information, we also achieve dynamic layer budgets. LAVa is the first unified
strategy for cache eviction and dynamic budget allocation that, unlike prior
methods, does not rely on training or the combination of multiple strategies.
Experiments with benchmarks (LongBench, Needle-In-A-Haystack, Ruler, and
InfiniteBench) demonstrate its superiority. Moreover, our experiments reveal a
new insight: dynamic layer budgets are crucial for generation tasks (e.g., code
completion), while dynamic head budgets play a key role in extraction tasks
(e.g., extractive QA). As a fully dynamic compression method, LAVa consistently
maintains top performance across task types. Our code is available at
https://github.com/MGDDestiny/Lava.
