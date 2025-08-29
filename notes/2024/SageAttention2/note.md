# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

> Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen

![111](../../blank.jpg)

## Abstract

Although quantization for linear layers has been widely used, its application
to accelerate the attention process remains limited. To further enhance the
efficiency of attention computation compared to SageAttention while maintaining
precision, we propose SageAttention2, which utilizes significantly faster 4-bit
matrix multiplication (Matmul) alongside additional precision-enhancing
techniques. First, we propose to quantize matrices $(Q, K)$ to INT4 in a
hardware-friendly thread-level granularity and quantize matrices $(\widetilde
P, V)$ to FP8. Second, we propose a method to smooth $Q$, enhancing the
accuracy of INT4 $QK^\top$. Third, we propose a two-level accumulation strategy
for $\widetilde PV$ to enhance the accuracy of FP8 $\widetilde PV$. The
operations per second (OPS) of SageAttention2 surpass FlashAttention2 and
xformers by about 3x and 4.5x on RTX4090, respectively. Moreover,
SageAttention2 matches the speed of FlashAttention3(fp8) on the Hopper GPUs,
while delivering much higher accuracy. Comprehensive experiments confirm that
our approach incurs negligible end-to-end metrics loss across diverse models,
including those for language, image, and video generation. The code is
available at https://github.com/thu-ml/SageAttention.
