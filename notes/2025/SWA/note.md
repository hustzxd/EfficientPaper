# Sliding Window Attention Training for Efficient Large Language Models

> Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao Wei, Tong Xu, Xiangyu Zhao

![111](cover.jpg)

## Abstract

Recent advances in transformer-based Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their quadratic computational complexity concerning sequence length remains a significant bottleneck for processing long documents. As a result, many efforts like sparse attention and state space models have been proposed to improve the efficiency of LLMs over long sequences. Though effective, these approaches compromise the performance or introduce structural complexity. This calls for a simple yet efficient model that preserves the fundamental Transformer architecture. To this end, we introduce SWAT, which enables efficient long-context handling via Sliding Window Attention Training. This paper first attributes the inefficiency of Transformers to the attention sink phenomenon resulting from the high variance of softmax operation. Then, we replace softmax with the sigmoid function and utilize a balanced ALiBi and Rotary Position Embedding for efficient information compression and retention. Experiments demonstrate that SWAT achieves SOTA performance compared with state-of-the-art linear recurrent architectures on eight benchmarks. Code is available at https://github.com/Fzkuji/swat-attention.


---

*以下总结由 MiMo 生成：*

这篇论文旨在解决Transformer模型处理长序列时计算复杂度高的问题。作者提出了SWAT方法，通过滑动窗口注意力训练，并用sigmoid函数替代softmax以缓解注意力沉降现象，同时结合平衡的ALiBi和旋转位置嵌入来提升信息压缩与保留效率。实验表明，SWAT在八个基准测试中达到了最先进性能，优于现有的线性递归架构。
