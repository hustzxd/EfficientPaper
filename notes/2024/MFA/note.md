# Multi-matrix Factorization Attention

> Jingcheng Hu, Houyi Li, Yinmin Zhang, Zili Wang, Shuigeng Zhou, Xiangyu Zhang, Heung-Yeung Shum, Daxin Jiang

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

We propose novel attention architectures, Multi-matrix Factorization
Attention (MFA) and MFA-Key-Reuse (MFA-KR). Existing variants for standard
Multi-Head Attention (MHA), including SOTA methods like MLA, fail to maintain
as strong performance under stringent Key-Value cache (KV cache) constraints.
MFA enhances model capacity by efficiently scaling up both the number and
dimension of attention heads through low-rank matrix factorization in the
Query-Key (QK) circuit. Extending MFA, MFA-KR further reduces memory
requirements by repurposing the key cache as value through value projection
re-parameterization. MFA's design enables strong model capacity when working
under tight KV cache budget, while MFA-KR is suitable for even harsher KV cache
limits with minor performance trade-off. Notably, in our extensive and
large-scale experiments, the proposed architecture outperforms MLA and performs
comparably to MHA, while reducing KV cache usage by up to 56% and 93.7%,
respectively.
