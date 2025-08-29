# Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs

![](../../blank.jpg)

## Abstract

To date, 2:4 sparsity has stood as the only sparse pattern that can be
accelerated using sparse tensor cores on GPUs. In practice, 2:4 sparsity often
possesses low actual speedups ($\leq 1.3$) and requires fixed sparse ratios,
meaning that other ratios, such as 4:8, 8:16, or those exceeding 50% sparsity,
do not incur any speedups on GPUs. Recent studies suggest that V:N:M sparsity
is promising in addressing these limitations of 2:4 sparsity. However,
regarding accuracy, the effects of V:N:M sparsity on broader Transformer
models, such as vision Transformers and large language models (LLMs), are
largely unexamined. Moreover, Some specific issues related to V:N:M sparsity,
such as how to select appropriate V and M values, remain unresolved. In this
study, we thoroughly investigate the application of V:N:M sparsity in vision
models and LLMs across multiple tasks, from pertaining to downstream tasks. We
propose three key approaches to enhance the applicability and accuracy of
V:N:M-sparse Transformers, including heuristic V and M selection,
V:N:M-specific channel permutation, and three-staged LoRA training techniques.
Experimental results show that, with our methods, the DeiT-small achieves
lossless accuracy at 64:2:5 sparsity, while the DeiT-base maintains accuracy
even at 64:2:8 sparsity. In addition, the fine-tuned LLama2-7B at 64:2:5
sparsity performs comparably or better than training-free 2:4 sparse
alternatives on downstream tasks. More importantly, V:N:M-sparse Transformers
offer a wider range of speedup-accuracy trade-offs compared to 2:4 sparsity.
Overall, our exploration largely facilitates the V:N:M sparsity to act as a
truly effective acceleration solution for Transformers in cost-sensitive
inference scenarios.
