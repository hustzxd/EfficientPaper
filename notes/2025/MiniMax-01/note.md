# MiniMax-01: Scaling Foundation Models with Lightning Attention

> MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu

![111](../../blank.jpg)

## Abstract

We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01,
which are comparable to top-tier models while offering superior capabilities in
processing longer contexts. The core lies in lightning attention and its
efficient scaling. To maximize computational capacity, we integrate it with
Mixture of Experts (MoE), creating a model with 32 experts and 456 billion
total parameters, of which 45.9 billion are activated for each token. We
develop an optimized parallel strategy and highly efficient
computation-communication overlap techniques for MoE and lightning attention.
This approach enables us to conduct efficient training and inference on models
with hundreds of billions of parameters across contexts spanning millions of
tokens. The context window of MiniMax-Text-01 can reach up to 1 million tokens
during training and extrapolate to 4 million tokens during inference at an
affordable cost. Our vision-language model, MiniMax-VL-01 is built through
continued training with 512 billion vision-language tokens. Experiments on both
standard and in-house benchmarks show that our models match the performance of
state-of-the-art models like GPT-4o and Claude-3.5-Sonnet while offering 20-32
times longer context window. We publicly release MiniMax-01 at
https://github.com/MiniMax-AI.
