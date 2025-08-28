# DReSS: Data-driven Regularized Structured Streamlining for Large Language Models

> Mingkuan Feng, Jinyang Wu, Shuai Zhang, Pengpeng Shao, Ruihan Jin, Zhengqi Wen, Jianhua Tao, Feihu Che

![111](fig1.png)

## Abstract

Large language models (LLMs) have achieved significant progress across
various domains, but their increasing scale results in high computational and
memory costs. Recent studies have revealed that LLMs exhibit sparsity,
providing the potential to reduce model size through pruning techniques.
However, existing pruning methods typically follow a prune-then-finetune
paradigm. Since the pruned components still contain valuable information, their
direct removal often leads to irreversible performance degradation, imposing a
substantial computational burden to recover performance during finetuning. In
this paper, we propose a novel paradigm that first applies regularization, then
prunes, and finally finetunes. Based on this paradigm, we introduce DReSS, a
simple and effective Data-driven Regularized Structured Streamlining method for
LLMs. By leveraging a small amount of data to regularize the components to be
pruned, DReSS explicitly transfers the important information to the remaining
parts of the model in advance. Compared to direct pruning, this can reduce the
information loss caused by parameter removal, thereby enhancing its language
modeling capabilities. Experimental results demonstrate that DReSS
significantly outperforms existing pruning methods even under extreme pruning
ratios, significantly reducing latency and increasing throughput.

- channel pruning mask直接初始化固定，通过正则化将mask的channel 信息传递到其他channel
- 根据 mask pruning
- 再次进行finetuning

step1的正则化会改变weight的分布，而且是按照人工设置的mask改变，是否会改变模型的基础能力呢？