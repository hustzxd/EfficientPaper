# OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization

![](../../blank.jpg)

## Abstract

Structured pruning is a promising approach for reducing the inference costs
of large vision and language models. By removing carefully chosen structures,
e.g., neurons or attention heads, the improvements from this approach can be
realized on standard deep learning hardware. In this work, we focus on
structured pruning in the one-shot (post-training) setting, which does not
require model retraining after pruning. We propose a novel combinatorial
optimization framework for this problem, based on a layer-wise reconstruction
objective and a careful reformulation that allows for scalable optimization.
Moreover, we design a new local combinatorial optimization algorithm, which
exploits low-rank updates for efficient local search. Our framework is time and
memory-efficient and considerably improves upon state-of-the-art one-shot
methods on vision models (e.g., ResNet50, MobileNet) and language models (e.g.,
OPT-1.3B -- OPT-30B). For language models, e.g., OPT-2.7B, OSSCAR can lead to
$125\times$ lower test perplexity on WikiText with $2\times$ inference time
speedup in comparison to the state-of-the-art ZipLM approach. Our framework is
also $6\times$ -- $8\times$ faster. Notably, our work considers models with
tens of billions of parameters, which is up to $100\times$ larger than what has
been previously considered in the structured pruning literature.

structured pruning的方式与之前方法相同，layer-by-layer优化每层的loss，loss由一阶和二阶信息构建。

如何搜索mask，从而最小化loss，是一个问题。首先是loss的计算比较耗时，论文中给出使用历史的计算和变化的mask位置来减少loss计算开销，这个也容易理解，只需要计算差值就好。因此，论文中给出一个local search的搜索方法。

文中没有说是否为unform pruning ratio，只给了加速比，通过代码发现每一层配置的pruning ratio是一致的。

$\hat{w}$