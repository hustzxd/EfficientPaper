# Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs?

> Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, Danqi Chen

![111](../../blank.jpg)

## Abstract

Language models handle increasingly long contexts for tasks such as book summarization, but this leads to growing memory costs for the key-value (KV) cache. Many prior works have proposed ways of discarding KVs from memory, but their approaches are tailored to favorable settings, obscuring caveats like high peak memory and performance degradation, and a fair comparison between methods is difficult. In this paper, we propose the *KV footprint* as a unified metric, which accounts for both the amount of KV entries stored and their lifespan in memory. We evaluate methods based on the smallest footprint they attain while preserving performance in both long-context understanding and generation, with context lengths of up to 128K tokens. This metric reveals the high peak memory of prior KV eviction methods. One class of methods -- *post-fill eviction* -- has a high footprint due to being incompatible with eviction during pre-filling. We adapt these methods to be able to evict KVs during pre-filling, achieving substantially lower KV footprints. We then turn to *recency eviction* methods, wherein we propose PruLong, an end-to-end optimization method for learning which attention heads need to retain the full KV cache and which do not. PruLong saves memory while preserving long-context performance, achieving 12% smaller KV footprint than prior methods while retaining performance in challenging recall tasks. Our paper clarifies the complex tangle of long-context inference methods and paves the way for future development to minimize the KV footprint.
