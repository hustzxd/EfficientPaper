# MiniCPM4: Ultra-Efficient LLMs on End Devices

> MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengdan Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Yukun Yan, Jiarui Yuan, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Ge Zhou, Jie Zhou, Wei Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun

<p align="center">
<img src="fig2.png" width="600" title="blank">
</p>

## Abstract

This paper introduces MiniCPM4, a highly efficient large language model (LLM)
designed explicitly for end-side devices. We achieve this efficiency through
systematic innovation in four key dimensions: model architecture, training
data, training algorithms, and inference systems. Specifically, in terms of
model architecture, we propose InfLLM v2, a trainable sparse attention
mechanism that accelerates both prefilling and decoding phases for long-context
processing. Regarding training data, we propose UltraClean, an efficient and
accurate pre-training data filtering and generation strategy, and UltraChat v2,
a comprehensive supervised fine-tuning dataset. These datasets enable
satisfactory model performance to be achieved using just 8 trillion training
tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient
pre-training strategy search, and improve existing post-training methods by
introducing chunk-wise rollout for load-balanced reinforcement learning and
data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose
CPM.cu that integrates sparse attention, model quantization, and speculative
sampling to achieve efficient prefilling and decoding. To meet diverse
on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B
parameters, respectively. Sufficient evaluation results show that MiniCPM4
outperforms open-source models of similar size across multiple benchmarks,
highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B
demonstrates significant speed improvements over Qwen3-8B when processing long
sequences. Through further adaptation, MiniCPM4 successfully powers diverse
applications, including trustworthy survey generation and tool use with model
context protocol, clearly showcasing its broad usability.
