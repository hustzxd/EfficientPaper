# SparseLLM: Towards Global Pruning for Pre-trained Language Models

![](../../blank.jpg)

## Abstract

The transformative impact of large language models (LLMs) like LLaMA and GPT
on natural language processing is countered by their prohibitive computational
demands. Pruning has emerged as a pivotal compression strategy, introducing
sparsity to enhance both memory and computational efficiency. Yet, traditional
global pruning is impractical for LLMs due to scalability issues, while local
pruning, despite its efficiency, leads to suboptimal solutions. Addressing
these challenges, we propose SparseLLM, a novel framework that redefines the
global pruning process into manageable, coordinated subproblems, allowing for
resource-efficient optimization with global optimality. SparseLLM's approach,
which conceptualizes LLMs as a chain of modular functions and leverages
auxiliary variables for problem decomposition, not only facilitates a pragmatic
application on LLMs but also demonstrates significant performance improvements,
particularly in high-sparsity regimes where it surpasses current
state-of-the-art methods.

之前sparsegpt，wanda等是按照layer-wise loss来逐层优化的，可以称之为 local pruning，通常只能得到次优解；相对而言，global pruning考虑全局的loss，从而在理论上能达到更好的效果。global pruning不可避免在会导致问题规模的扩大，该工作考虑切分为多个subproblem来缓解；

global pruning使用较少的数据集时也有overfit的风险！！

结果提升不大