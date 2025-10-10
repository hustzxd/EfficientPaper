# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

> Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin

![111](fig1.png)

## Abstract

Gating mechanisms have been widely utilized, from early models like LSTMs and
Highway Networks to recent state space models, linear attention, and also
softmax attention. Yet, existing literature rarely examines the specific
effects of gating. In this work, we conduct comprehensive experiments to
systematically investigate gating-augmented softmax attention variants.
Specifically, we perform a comprehensive comparison over 30 variants of 15B
Mixture-of-Experts (MoE) models and 1.7B dense models trained on a 3.5 trillion
token dataset. Our central finding is that a simple modification-applying a
head-specific sigmoid gate after the Scaled Dot-Product Attention
(SDPA)-consistently improves performance. This modification also enhances
training stability, tolerates larger learning rates, and improves scaling
properties. By comparing various gating positions and computational variants,
we attribute this effectiveness to two key factors: (1) introducing
non-linearity upon the low-rank mapping in the softmax attention, and (2)
applying query-dependent sparse gating scores to modulate the SDPA output.
Notably, we find this sparse gating mechanism mitigates 'attention sink' and
enhances long-context extrapolation performance, and we also release related
$\href{https://github.com/qiuzh20/gated_attention}{codes}$ and
$\href{https://huggingface.co/QwQZh/gated_attention}{models}$ to facilitate
future research.
