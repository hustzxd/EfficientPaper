# Recycled Attention: Efficient inference for long-context language models

![](fig1.png)

## Abstract

Generating long sequences of tokens given a long-context input imposes a
heavy computational burden for large language models (LLMs). One of the
computational bottleneck comes from computing attention over a long sequence of
input at each generation step. In this paper, we propose Recycled Attention, an
inference-time method which alternates between full context attention and
attention over a subset of input tokens. When performing partial attention, we
recycle the attention pattern of a previous token that has performed full
attention and attend only to the top K most attended tokens, reducing the cost
of data movement and attention computation. Compared to previously proposed
inference-time acceleration method which attends only to local context or
tokens with high accumulative attention scores, our approach flexibly chooses
tokens that are relevant to the current decoding step. We evaluate our methods
on RULER, a suite of tasks designed to comprehensively evaluate long-context
abilities, and long-context language modeling tasks. Applying our method to
off-the-shelf LLMs achieves comparable speedup to baselines which only consider
local context while improving the performance by 2x. We further explore two
ideas to improve performance-efficiency trade-offs: (1) dynamically decide when
to perform recycled or full attention step based on the query similarities and
(2) continued pre-training the model with Recycled Attention.

方法比较简单，ICLR审稿意见也说方法创新性低
https://openreview.net/forum?id=8qYuxV4lRu