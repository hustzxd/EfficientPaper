# Pruning General Large Language Models into Customized Expert Models

![](fig1.png)

## Abstract

Large language models (LLMs) have revolutionized natural language processing,
yet their substantial model sizes often require substantial computational
resources. To preserve computing resources and accelerate inference speed, it
is crucial to prune redundant parameters, especially for experienced users who
often need compact expert models tailored to specific downstream scenarios.
However, most existing pruning methods focus on preserving the model's general
capabilities, often requiring extensive post-training or suffering from
degraded performance due to coarse-grained pruning. In this work, we design a
$\underline{Cus}$tom $\underline{Prun}$ing method ($\texttt{Cus-Prun}$) to
prune a large general model into a smaller lightweight expert model, which is
positioned along the "language", "domain" and "task" dimensions. By identifying
and pruning irrelevant neurons of each dimension, $\texttt{Cus-Prun}$ creates
expert models without any post-training. Our experiments demonstrate that
$\texttt{Cus-Prun}$ consistently outperforms other methods, achieving minimal
loss in both expert and general capabilities across various models from
different model families and sizes.

language domain task 三个维度对模型进行结构化稀疏，然后按照实际的任务，将三个维度的mask进行融合，得到最终的mask。