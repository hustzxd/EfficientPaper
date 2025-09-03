# Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models

> Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, Yiran Zhong

![111](../../blank.jpg)

## Abstract

Linear attention is an efficient attention mechanism that has recently
emerged as a promising alternative to conventional softmax attention. With its
ability to process tokens in linear computational complexities, linear
attention, in theory, can handle sequences of unlimited length without
sacrificing speed, i.e., maintaining a constant training speed for various
sequence lengths with a fixed memory consumption. However, due to the issue
with cumulative summation (cumsum), current linear attention algorithms cannot
demonstrate their theoretical advantage in a causal setting. In this paper, we
present Lightning Attention-2, the first linear attention implementation that
enables linear attention to realize its theoretical computational benefits. To
achieve this, we leverage the thought of tiling, separately handling the
intra-block and inter-block components in linear attention calculation.
Specifically, we utilize the conventional attention computation mechanism for
the intra-blocks and apply linear attention kernel tricks for the inter-blocks.
A tiling technique is adopted through both forward and backward procedures to
take full advantage of the GPU hardware. We implement our algorithm in Triton
to make it IO-aware and hardware-friendly. Various experiments are conducted on
different model sizes and sequence lengths. Lightning Attention-2 retains
consistent training and inference speed regardless of input sequence length and
is significantly faster than other attention mechanisms. The source code is
available at https://github.com/OpenNLPLab/lightning-attention.
