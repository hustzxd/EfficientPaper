# Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts

> Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, Quan Chen, Xin Liu

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Mixture-of-experts (MoE) has been extensively employed to scale large
language models to trillion-plus parameters while maintaining a fixed
computational cost. The development of large MoE models in the distributed
scenario encounters the problem of large communication overhead. The
inter-device communication of a MoE layer can occupy 47% time of the entire
model execution with popular models and frameworks. Therefore, existing methods
suggest the communication in a MoE layer to be pipelined with the computation
for overlapping. However, these coarse grained overlapping schemes introduce a
notable impairment of computational efficiency and the latency concealing is
sub-optimal.
  To this end, we present COMET, an optimized MoE system with fine-grained
communication-computation overlapping. Leveraging data dependency analysis and
task rescheduling, COMET achieves precise fine-grained overlapping of
communication and computation. Through adaptive workload assignment, COMET
effectively eliminates fine-grained communication bottlenecks and enhances its
adaptability across various scenarios. Our evaluation shows that COMET
accelerates the execution of a single MoE layer by $1.96\times$ and for
end-to-end execution, COMET delivers a $1.71\times$ speedup on average. COMET
has been adopted in the production environment of clusters with
ten-thousand-scale of GPUs, achieving savings of millions of GPU hours.
