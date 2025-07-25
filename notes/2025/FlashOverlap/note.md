# FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation

> Ke Hong, Xiuhong Li, Minxu Liu, Qiuli Mao, Tianqi Wu, Zixiao Huang, Lufang Chen, Zhong Wang, Yichong Zhang, Zhenhua Zhu, Guohao Dai, Yu Wang

<p align="center">
<img src="fig3.png" width="600" title="blank">
</p>

## Abstract

Generative models have achieved remarkable success across various
applications, driving the demand for multi-GPU computing. Inter-GPU
communication becomes a bottleneck in multi-GPU computing systems, particularly
on consumer-grade GPUs. By exploiting concurrent hardware execution,
overlapping computation and communication latency is an effective technique for
mitigating the communication overhead. We identify that an efficient and
adaptable overlapping design should satisfy (1) tile-wise overlapping to
maximize the overlapping opportunity, (2) interference-free computation to
maintain the original computational performance, and (3) communication
agnosticism to reduce the development burden against varying communication
primitives. Nevertheless, current designs fail to simultaneously optimize for
all of those features.
  To address the issue, we propose FlashOverlap, a lightweight design
characterized by tile-wise overlapping, interference-free computation, and
communication agnosticism. FlashOverlap utilizes a novel signaling mechanism to
identify tile-wise data dependency without interrupting the computation
process, and reorders data to contiguous addresses, enabling communication by
simply calling NCCL APIs. Experiments show that such a lightweight design
achieves up to 1.65x speedup, outperforming existing works in most cases.
