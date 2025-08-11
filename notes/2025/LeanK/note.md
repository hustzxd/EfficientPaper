# LeanK: Learnable K Cache Channel Pruning for Efficient Decoding

> Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Large language models (LLMs) enable long-context tasks but face efficiency
challenges due to the growing key-value (KV) cache. We propose LeanK, a
learning-based method that prunes unimportant key (K) cache channels by
leveraging static channel sparsity. With a novel two-stage training process,
LeanK learns channel-wise static mask that could satisfy specific sparsity
ratio and hardware alignment requirement. LeanK reduces GPU memory and
accelerates decoding without sacrificing accuracy. Experiments demonstrate up
to 70% K cache and 16%-18% V cache memory reduction. Custom decoding kernel
enables 1.3x speedup for attention computation. We also provide insights into
model channels and attention heads during long-context inference by analyzing
the learned importance distribution. Our code is available at
https://aka.ms/LeanK.
