# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs

> Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, Zhicheng Liu

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

We present Pangu Ultra, a Large Language Model (LLM) with 135 billion
parameters and dense Transformer modules trained on Ascend Neural Processing
Units (NPUs). Although the field of LLM has been witnessing unprecedented
advances in pushing the scale and capability of LLM in recent years, training
such a large-scale model still involves significant optimization and system
challenges. To stabilize the training process, we propose depth-scaled sandwich
normalization, which effectively eliminates loss spikes during the training
process of deep models. We pre-train our model on 13.2 trillion diverse and
high-quality tokens and further enhance its reasoning capabilities during
post-training. To perform such large-scale training efficiently, we utilize
8,192 Ascend NPUs with a series of system optimizations. Evaluations on
multiple diverse benchmarks indicate that Pangu Ultra significantly advances
the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral
Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse
model structure contains much more parameters. Our exploration demonstrates
that Ascend NPUs are capable of efficiently and effectively training dense
models with more than 100 billion parameters. Our model and system will be
available for our commercial customers.
