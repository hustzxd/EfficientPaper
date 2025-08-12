# Transformer Acceleration with Dynamic Sparse Attention

> Liu Liu, Zheng Qu, Zhaodong Chen, Yufei Ding, Yuan Xie

![](fig2.png)

## Abstract

Transformers are the mainstream of NLP applications and are becoming
increasingly popular in other domains such as Computer Vision. Despite the
improvements in model quality, the enormous computation costs make Transformers
difficult at deployment, especially when the sequence length is large in
emerging applications. Processing attention mechanism as the essential
component of Transformer is the bottleneck of execution due to the quadratic
complexity. Prior art explores sparse patterns in attention to support long
sequence modeling, but those pieces of work are on static or fixed patterns. We
demonstrate that the sparse patterns are dynamic, depending on input sequences.
Thus, we propose the Dynamic Sparse Attention (DSA) that can efficiently
exploit the dynamic sparsity in the attention of Transformers. Compared with
other methods, our approach can achieve better trade-offs between accuracy and
model complexity. Moving forward, we identify challenges and provide solutions
to implement DSA on existing hardware (GPUs) and specialized hardware in order
to achieve practical speedup and efficiency improvements for Transformer
execution.

需要训练，seerattention和这个论文思路非常像。