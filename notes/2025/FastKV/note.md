# FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

<p align="center">
<img src="fig2.png" width="600" title="blank">
</p>

## Abstract

While large language models (LLMs) excel at handling long-context sequences,
they require substantial key-value (KV) caches to store contextual information,
which can heavily burden computational efficiency and memory usage. Previous
efforts to compress these KV caches primarily focused on reducing memory
demands but were limited in enhancing latency. To address this issue, we
introduce FastKV, a KV cache compression method designed to enhance latency for
long-context sequences. To enhance processing speeds while maintaining
accuracy, FastKV adopts a novel Token-Selective Propagation (TSP) approach that
retains the full context information in the initial layers of LLMs and
selectively propagates only a portion of this information in deeper layers even
in the prefill stage. Additionally, FastKV incorporates grouped-query attention
(GQA)-aware KV cache compression to exploit the advantages of GQA in both
memory and computational efficiency. Our experimental results show that FastKV
achieves 2.00$\times$ and 1.40$\times$ improvements in time-to-first-token
(TTFT) and throughput, respectively, compared to HeadKV, the state-of-the-art
KV cache compression method. Moreover, FastKV successfully maintains accuracy
on long-context benchmarks at levels comparable to the baselines. Our code is
available at https://github.com/dongwonjo/FastKV.

