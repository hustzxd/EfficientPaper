# Unveiling Super Experts in Mixture-of-Experts Large Language Models

> Zunhai Su, Qingyuan Li, Hao Zhang, YuLei Qian, Yuchen Xie, Kehong Yuan

<p align="center">
<img src="fig3.png" width="600" title="blank">
</p>

## Abstract

Sparsely activated Mixture-of-Experts (MoE) models have shown promise in
enhancing the learning capacity of large language models (LLMs). Leveraging the
intrinsic importance differences among experts, recent research has explored
expert-level compression techniques to improve the efficiency of MoE LLMs.
However, existing approaches often rely on empirical criteria to identify
critical experts, lacking a deeper exploration and understanding of the
heterogeneous importance of experts. In this study, we present the first
discovery and investigation of a distinct subset of experts that play a crucial
role in the underlying mechanisms during the model's forward inference. These
experts are prevalent in open-source MoE LLMs, and despite their limited
number, pruning them leads to a significant decline in model performance (e.g.,
pruning three causes Qwen3-30B-A3B to produce repetitive and uninformative
outputs). We refer to these experts as Super Experts (SEs). Our comprehensive
analysis provides progressively deeper insights into SEs. (i) SEs are
characterized by rare but extreme activation outliers in the output of the
down_proj, which give rise to massive activations in the hidden states between
decoder layers. Moreover, the distribution of SEs remains model-specific and is
unaffected by post-training processes. (ii) By pruning SEs, we assess their
significance across a variety of tasks, revealing their considerable impact on
the model's overall performance, particularly in mathematical reasoning. (iii)
We further enhance our understanding of the influence of SEs compression. Our
findings confirm that MoE LLMs rely on SEs to induce attention sinks, which are
crucial for the distribution of attention scores but are significantly
disrupted by SE pruning. The code is available at
https://github.com/ZunhaiSu/Super-Experts-Profilling.
