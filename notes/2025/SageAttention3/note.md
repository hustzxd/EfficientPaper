# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

> Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen

![111](../../blank.jpg)

## Abstract

The efficiency of attention is important due to its quadratic time
complexity. We enhance the efficiency of attention through two key
contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to
accelerate attention computation. Our implementation achieves 1038 TOPS on
RTX5090, which is a 5x speedup over the fastest FlashAttention on RTX5090.
Experiments show that our FP4 attention can accelerate inference of various
models in a plug-and-play way. Second, we pioneer low-bit attention to training
tasks. Existing low-bit attention works like FlashAttention3 and SageAttention
focus only on inference. However, the efficiency of training large models is
also important. To explore whether low-bit attention can be effectively applied
to training tasks, we design an accurate and efficient 8-bit attention for both
forward and backward propagation. Experiments indicate that 8-bit attention
achieves lossless performance in fine-tuning tasks but exhibits slower
convergence in pretraining tasks. The code will be available at
https://github.com/thu-ml/SageAttention.
