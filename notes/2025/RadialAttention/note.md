# Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation

> Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, Song Han

![](../../blank.jpg)

## Abstract

Recent advances in diffusion models have enabled high-quality video
generation, but the additional temporal dimension significantly increases
computational costs, making training and inference on long videos prohibitively
expensive. In this paper, we identify a phenomenon we term Spatiotemporal
Energy Decay in video diffusion models: post-softmax attention scores diminish
as spatial and temporal distance between tokens increase, akin to the physical
decay of signal or waves over space and time in nature. Motivated by this, we
propose Radial Attention, a scalable sparse attention mechanism with $O(n \log
n)$ complexity that translates energy decay into exponentially decaying compute
density, which is significantly more efficient than standard $O(n^2)$ dense
attention and more expressive than linear attention. Specifically, Radial
Attention employs a simple, static attention mask where each token attends to
spatially nearby tokens, with the attention window size shrinking with temporal
distance. Moreover, it allows pre-trained video diffusion models to extend
their generation length with efficient LoRA-based fine-tuning. Extensive
experiments show that Radial Attention maintains video quality across
Wan2.1-14B, HunyuanVideo, and Mochi 1, achieving up to a 1.9$\times$ speedup
over the original dense attention. With minimal tuning, it enables video
generation up to 4$\times$ longer while reducing training costs by up to
4.4$\times$ compared to direct fine-tuning and accelerating inference by up to
3.7$\times$ compared to dense attention inference.
