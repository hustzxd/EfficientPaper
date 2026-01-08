# NOSA: Native and Offloadable Sparse Attention

> Yuxiang Huang, Chaojun Xiao, Xu Han, Zhiyuan Liu

![111](../../blank.jpg)

## Abstract

Trainable sparse attention has emerged as a promising solution to address the decoding efficiency bottleneck of LLMs in long-context processing, significantly saving memory accesses while minimally impacting task performance. However, existing sparse attention methods leave a crucial limitation unresolved: the size of the key-value (KV) cache remains unreduced, which constrains on-GPU batch sizes and throttles decoding throughput, especially in large-scale batched inference. In this paper, we show that trainable sparse attention naturally exhibits strong locality in token selection across adjacent decoding steps, thereby enabling KV cache offloading without altering the underlying attention computation. However, the inherent locality remains insufficient to achieve efficient offloading, as the transfer of selected KV pairs between the CPU and GPU continues to dominate the overall decoding cost. Building on this insight, we present NOSA, a trainable sparse attention framework designed to natively support KV cache offloading. NOSA introduces explicit locality constraints by decomposing token selection into query-aware and query-agnostic components, thereby reducing KV transfers while preserving the same attention computation as used during training. We pretrain a 1B-parameter model with NOSA and conduct extensive benchmarks, showing that it preserves near-lossless performance while achieving up to a 2.3x improvement in decoding throughput compared with the vanilla trainable sparse attention baseline (InfLLM-V2).
