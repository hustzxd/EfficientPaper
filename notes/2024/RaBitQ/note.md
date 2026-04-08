# RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search

> Jianyang Gao, Cheng Long

![111](../../blank.jpg)

## Abstract

Searching for approximate nearest neighbors (ANN) in the high-dimensional Euclidean space is a pivotal problem. Recently, with the help of fast SIMD-based implementations, Product Quantization (PQ) and its variants can often efficiently and accurately estimate the distances between the vectors and have achieved great success in the in-memory ANN search. Despite their empirical success, we note that these methods do not have a theoretical error bound and are observed to fail disastrously on some real-world datasets. Motivated by this, we propose a new randomized quantization method named RaBitQ, which quantizes $D$-dimensional vectors into $D$-bit strings. RaBitQ guarantees a sharp theoretical error bound and provides good empirical accuracy at the same time. In addition, we introduce efficient implementations of RaBitQ, supporting to estimate the distances with bitwise operations or SIMD-based operations. Extensive experiments on real-world datasets confirm that (1) our method outperforms PQ and its variants in terms of accuracy-efficiency trade-off by a clear margin and (2) its empirical performance is well-aligned with our theoretical analysis.


---

*以下总结由 MiMo 生成：*

这篇论文旨在解决高维向量近似最近邻搜索中现有量化方法缺乏理论误差界且在某些真实数据集上表现不佳的问题。作者提出了一种名为RaBitQ的随机量化方法，将D维向量量化为D位字符串，并提供了严格的理论误差保证。RaBitQ同时支持基于位运算或SIMD的高效距离估计实现。实验表明，该方法在精度-效率权衡上显著优于乘积量化（PQ）及其变体，且其经验性能与理论分析高度一致。
