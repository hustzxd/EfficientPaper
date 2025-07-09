# FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion

> Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, Haibin Lin, Xin Jin, Xin Liu

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Large deep learning models have demonstrated strong ability to solve many
tasks across a wide range of applications. Those large models typically require
training and inference to be distributed. Tensor parallelism is a common
technique partitioning computation of an operation or layer across devices to
overcome the memory capacity limitation of a single processor, and/or to
accelerate computation to meet a certain latency requirement. However, this
kind of parallelism introduces additional communication that might contribute a
significant portion of overall runtime. Thus limits scalability of this
technique within a group of devices with high speed interconnects, such as GPUs
with NVLinks in a node. This paper proposes a novel method, Flux, to
significantly hide communication latencies with dependent computations for
GPUs. Flux over-decomposes communication and computation operations into much
finer-grained operations and further fuses them into a larger kernel to
effectively hide communication without compromising kernel efficiency. Flux can
potentially overlap up to 96% of communication given a fused kernel. Overall,
it can achieve up to 1.24x speedups for training over Megatron-LM on a cluster
of 128 GPUs with various GPU generations and interconnects, and up to 1.66x and
1.30x speedups for prefill and decoding inference over vLLM on a cluster with 8
GPUs with various GPU generations and interconnects.
