# HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

> Ping Gong, Jiawei Yi, Shengnan Wang, Juncheng Zhang, Zewen Jin, Ouxiang Zhou, Ruibo Liu, Guanbin Xu, Youhui Bai, Bowen Ye, Kun Yuan, Tong Yang, Gong Zhang, Renhai Chen, Feng Wu, Cheng Li

![](fig2.png)

## Abstract

Large Language Models (LLMs) have emerged as a pivotal research area, yet the
attention module remains a critical bottleneck in LLM inference, even with
techniques like KVCache to mitigate redundant computations. While various
top-$k$ attention mechanisms have been proposed to accelerate LLM inference by
exploiting the inherent sparsity of attention, they often struggled to strike a
balance between efficiency and accuracy. In this paper, we introduce HATA
(Hash-Aware Top-$k$ Attention), a novel approach that systematically integrates
low-overhead learning-to-hash techniques into the Top-$k$ attention process.
Different from the existing top-k attention methods which are devoted to
seeking an absolute estimation of qk score, typically with a great cost, HATA
maps queries and keys into binary hash codes, and acquires the relative qk
score order with a quite low cost, which is sufficient for realizing top-k
attention. Extensive experiments demonstrate that HATA achieves up to
7.2$\times$ speedup compared to vanilla full attention while maintaining model
accuracy. In addition, HATA outperforms the state-of-the-art top-$k$ attention
methods in both accuracy and efficiency across multiple mainstream LLM models
and diverse tasks. HATA is open source at https://github.com/gpzlx1/HATA.

HATA用于decoding时的加速，之前kv压缩的方法，比如snapkv用历史的结果预测当前结果，quest按照block pool预测block的重要性，seerattention通过训练的方法来预测重要性；

HATA提出将QK映射为Hash Code，并实用xor 操作高效的判断K cache的重要性。

映射Hash Code的过程涉及到可学习权重，需要一部分数据集进行学习

和 HashAttention: Semantic Sparsity for Faster Inference 思路很像