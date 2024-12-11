# Enhancing One-shot Pruned Pre-trained Language Models through Sparse-Dense-Sparse Mechanism

<p align="center">
<img src="sds.png" width="600" title="blank">
</p>

## Abstract

Pre-trained language models (PLMs) are engineered to be robust in contextual
understanding and exhibit outstanding performance in various natural language
processing tasks. However, their considerable size incurs significant
computational and storage costs. Modern pruning strategies employ one-shot
techniques to compress PLMs without the need for retraining on task-specific or
otherwise general data; however, these approaches often lead to an
indispensable reduction in performance. In this paper, we propose SDS, a
Sparse-Dense-Sparse pruning framework to enhance the performance of the pruned
PLMs from a weight distribution optimization perspective. We outline the
pruning process in three steps. Initially, we prune less critical connections
in the model using conventional one-shot pruning methods. Next, we reconstruct
a dense model featuring a pruning-friendly weight distribution by reactivating
pruned connections with sparse regularization. Finally, we perform a second
pruning round, yielding a superior pruned model compared to the initial
pruning. Experimental results demonstrate that SDS outperforms the
state-of-the-art pruning techniques SparseGPT and Wanda under an identical
sparsity configuration. For instance, SDS reduces perplexity by 9.13 on
Raw-Wikitext2 and improves accuracy by an average of 2.05% across multiple
zero-shot benchmarks for OPT-125M with 2:4 sparsity.
