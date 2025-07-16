# TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives

> Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Xin Liu

<p align="center">
<img src="fig7.png" width="600" title="blank">
</p>

## Abstract

Large deep learning models have achieved state-of-the-art performance in a
wide range of tasks. These models often necessitate distributed systems for
efficient training and inference. The fundamental building blocks for
distributed model execution are intra-layer parallel operators. The most
effective approach to enhancing the performance of intra-layer parallel
operators involves overlapping computation with communication. The overlapping
can be achieved through either operator decomposition or kernel fusion. While
decomposing operators is straightforward to implement, it often results in
suboptimal performance. On the other hand, fusing communication kernels with
compute kernels demands significant expertise and is error-prone.
  In this paper, we propose TileLink to enable efficient compilation and
generation of overlapped compute-communication kernels. TileLink is composed of
frontend and backend. In the frontend, TileLink decouples the design space of
communication and computation, linking these two parts via tile-centric
primitives. In the backend, TileLink translates these primitives into low-level
communication instructions, integrating the communication and computation
components to achieve overlapped execution. In experiments, TileLink achieves
from $1.17\times$ to $20.76\times$ speedup to non-overlapping baseline and
achieves performance comparable to state-of-the-art overlapping libraries on
GPUs.
