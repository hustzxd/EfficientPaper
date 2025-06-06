# Hardware-Efficient Attention for Fast Decoding

<p align="center">
<img src="gla.png" width="600" title="blank">
</p>

## Abstract

LLM decoding is bottlenecked for large batches and long contexts by loading
the key-value (KV) cache from high-bandwidth memory, which inflates per-token
latency, while the sequential nature of decoding limits parallelism. We analyze
the interplay among arithmetic intensity, parallelization, and model quality
and question whether current architectures fully exploit modern hardware. This
work redesigns attention to perform more computation per byte loaded from
memory to maximize hardware efficiency without trading off parallel
scalability. We first propose Grouped-Tied Attention (GTA), a simple variant
that combines and reuses key and value states, reducing memory transfers
without compromising model quality. We then introduce Grouped Latent Attention
(GLA), a parallel-friendly latent attention paired with low-level optimizations
for fast decoding while maintaining high model quality. Experiments show that
GTA matches Grouped-Query Attention (GQA) quality while using roughly half the
KV cache and that GLA matches Multi-head Latent Attention (MLA) and is easier
to shard. Our optimized GLA kernel is up to 2$\times$ faster than FlashMLA, for
example, in a speculative decoding setting when the query length exceeds one.
Furthermore, by fetching a smaller KV cache per device, GLA reduces end-to-end
latency and increases throughput in online serving benchmarks by up to
2$\times$.


MLA的改进，将latent划分group，类似MHA到GQA的改进。