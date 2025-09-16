# Faster VGGT with Block-Sparse Global Attention

> Chung-Shien Brian Wang, Christian Schmidt, Jens Piekenbrinck, Bastian Leibe

![111](../../blank.jpg)

## Abstract

Efficient and accurate feed-forward multi-view reconstruction has long been
an important task in computer vision. Recent transformer-based models like VGGT
and $\pi^3$ have achieved impressive results with simple architectures, yet
they face an inherent runtime bottleneck, due to the quadratic complexity of
the global attention layers, that limits the scalability to large image sets.
In this paper, we empirically analyze the global attention matrix of these
models and observe that probability mass concentrates on a small subset of
patch-patch interactions that correspond to cross-view geometric matches.
Motivated by the structured attention and inspired by recent advancement in
large language models, we propose a replacement for the dense global attention
operation based on highly optimized block-sparse kernels, yielding up to
$4\times$ faster inference with comparable task performance. Our retrofit
requires no retraining of the backbone, extends to both VGGT and $\pi^3$, and
supports large image collections. Evaluations on a comprehensive suite of
multi-view benchmarks demonstrate the effectiveness of our approach.
