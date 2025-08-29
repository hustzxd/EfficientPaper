# MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving

> Ruihao Li, Shagnik Pal, Vineeth Narayan Pullu, Prasoon Sinha, Jeeho Ryoo, Lizy K. John, Neeraja J. Yadwadkar

![](../../blank.jpg)

## Abstract

KV cache accelerates LLM inference by avoiding redundant computation, at the
expense of memory. To support larger KV caches, prior work extends GPU memory
with CPU memory via CPU-offloading. This involves swapping KV cache between GPU
and CPU memory. However, because the cache updates dynamically, such swapping
incurs high CPU memory traffic. We make a key observation that model parameters
remain constant during runtime, unlike the dynamically updated KV cache.
Building on this, we introduce MIRAGE, which avoids KV cache swapping by
remapping, and thereby repurposing, the memory allocated to model parameters
for KV cache. This parameter remapping is especially beneficial in multi-tenant
environments, where the memory used for the parameters of the inactive models
can be more aggressively reclaimed. Exploiting the high CPU-GPU bandwidth
offered by the modern hardware, such as the NVIDIA Grace Hopper Superchip, we
show that MIRAGE significantly outperforms state-of-the-art solutions,
achieving a reduction of 44.8%-82.5% in tail time-between-token latency,
20.7%-99.3% in tail time-to-first-token latency, and 6.6%-86.7% higher
throughput compared to vLLM.
