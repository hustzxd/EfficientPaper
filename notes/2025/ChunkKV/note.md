# ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

To reduce memory costs in long-context inference with Large Language Models
(LLMs), many recent works focus on compressing the key-value (KV) cache of
different tokens. However, we identify that the previous KV cache compression
methods measure token importance individually, neglecting the dependency
between different tokens in the real-world language characterics. In light of
this, we introduce ChunkKV, grouping the tokens in a chunk as a basic
compressing unit, and retaining the most informative semantic chunks while
discarding the less important ones. Furthermore, observing that ChunkKV
exhibits higher similarity in the preserved indices across different layers, we
propose layer-wise index reuse to further reduce computational overhead. We
evaluated ChunkKV on cutting-edge long-context benchmarks including LongBench
and Needle-In-A-HayStack, as well as the GSM8K and JailbreakV in-context
learning benchmark. Our experiments with instruction tuning and multi-step
reasoning (O1 and R1) LLMs, achieve up to 10\% performance improvement under
aggressive compression ratios compared to existing methods.

按照chunk来判断重要性，而不是每个token，这样能更好的保留tokens间的语义信息。