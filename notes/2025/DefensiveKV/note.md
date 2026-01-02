# Taming the Fragility of KV Cache Eviction in LLM Inference

> Yuan Feng, Haoyu Guo, JunLin Lv, S. Kevin Zhou, Xike Xie

![111](../../blank.jpg)

## Abstract

Large language models have revolutionized natural language processing, yet their deployment remains hampered by the substantial memory and runtime overhead of the transformer's Key-Value cache. To mitigate this, recent methods employ a scoring-aggregation framework to evict unimportant cache entries, based on the stability assumption-that a fixed subset of entries remains consistently important during generation. However, prior work has largely focused on refining importance indicators for scoring, while defaulting to mean aggregation due to a faithful trust in the stability assumption. In this work, we argue that this underlying assumption is inherently fragile, making mean aggregation highly vulnerable in extreme cases. To counter this, we propose a simple yet elegant defensive aggregation strategy: a two-step, linear-time approach that controls worst-case risk, thereby defending against extreme cases with negligible computational overhead. Embodying this strategy, we propose a novel cache eviction method, DefensiveKV and its extension, Layer-DefensiveKV, which incorporates layer-wise budget allocation. Across seven task domains (18 datasets), our methods reduce generation quality loss by 2.3x and 4.3x respectively, versus the strongest baseline under a 20% cache size. These results set new performance benchmarks and pioneer a promising direction for optimizing cache eviction against underlying fragility through worst-case risk management. Our code is available at https://github.com/FFY0/DefensiveKV.
