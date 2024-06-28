# Sparse-IFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Recent research has focused on weight sparsity in neural network training to
reduce FLOPs, aiming for improved efficiency (test accuracy w.r.t training
FLOPs). However, sparse weight training often sacrifices accuracy, requiring
extended training schedules to attain the accuracy of dense models. In
contrast, our approach, Sparse Iso-FLOP Transformations (Sparse-IFT), uses
sparsity to improve accuracy while maintaining dense model FLOPs. Using a
single hyperparameter (i.e., sparsity level), Sparse-IFTs efficiently replace
dense layers, expanding the search space for optimal sparse masks. In addition,
dynamic sparse training with Sparse-IFT models effectively navigates this
larger sparse mask-weight space, which is evidenced by a spectral analysis
using Ramanujan graph properties. Our study reveals a robust correlation among
mask topology, weights, and final performance. Notably, without adjusting
hyperparameters, replacing dense layers with Sparse-IFT yields significant
improvements, such as a +3.5% boost for ResNet-18 on ImageNet and +0.9% for
GPT-3 Small on the Open LLM leaderboard. To our knowledge, this is the first
work to demonstrate the use of sparsity for improving the accuracy of dense
models through a simple-to-use set of sparse transformations. Code is available
at: https://github.com/CerebrasResearch/Sparse-IFT.
