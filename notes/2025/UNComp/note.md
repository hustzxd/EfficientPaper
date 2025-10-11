# UNComp: Can Matrix Entropy Uncover Sparsity? -- A Compressor Design from an Uncertainty-Aware Perspective

> Jing Xiong, Jianghan Shen, Fanghua Ye, Chaofan Tao, Zhongwei Wan, Jianqiao Lu, Xun Wu, Chuanyang Zheng, Zhijiang Guo, Min Yang, Lingpeng Kong, Ngai Wong

![111](fig4.png)

## Abstract

Deploying large language models (LLMs) for long-context inference remains
challenging due to their substantial memory and computational demands. While
techniques such as Key-Value (KV) cache compression are designed to reduce
memory usage, they often neglect the structured sparsity inherent in the
relationship between hidden states and their corresponding KV cache. In this
work, we explore the role of uncertainty as a potential indicator of sparsity
within LLMs. We propose UNComp, an uncertainty-aware framework that leverages
truncated matrix entropy to identify areas of low information content, thereby
revealing sparsity patterns that can be used for adaptive compression. Unlike
traditional methods that apply uniform compression, UNComp dynamically adjusts
its approach to compression, guided by uncertainty measures that reflect the
importance of various model components. Our analysis shows that sparsity
patterns, when derived from uncertainty estimates, can be exploited to reveal
special long-range dependencies, such as retrieval heads and retrieval layers.
This perspective not only enhances our understanding of how compression can be
optimized but also provides new insights into the inherent sparsity of LLMs
during long-context inference. By focusing on uncertainty to analyze the
sparsity pattern in detail, UNComp reduces the KV cache size to 4.74% of the
original, achieves a 6% prefill speedup, and improves throughput by 6.4x - not
only delivering strong lossless compression performance, but also validating
the effectiveness of the underlying theoretical tool. We release the code at
https://github.com/menik1126/UNComp.
