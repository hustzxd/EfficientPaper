# SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting

<p align="center">
<img src="fig9.png" width="600" title="blank">
</p>

## Abstract

Early exiting has recently emerged as a promising technique for accelerating
large language models (LLMs) by effectively reducing the hardware computation
and memory access. In this paper, we present SpecEE, a fast LLM inference
engine with speculative early exiting. (1) At the algorithm level, we propose
the speculation-based lightweight predictor design by exploiting the
probabilistic correlation between the speculative tokens and the correct
results and high parallelism of GPUs. (2) At the system level, we point out
that not all layers need a predictor and design the two-level heuristic
predictor scheduling engine based on skewed distribution and contextual
similarity. (3) At the mapping level, we point out that different decoding
methods share the same essential characteristics, and propose the context-aware
merged mapping for predictor with efficient GPU implementations to support
speculative decoding, and form a framework for various existing orthogonal
acceleration techniques (e.g., quantization and sparse activation) on cloud and
personal computer (PC) scenarios, successfully pushing the Pareto frontier of
accuracy and speedup. It is worth noting that SpecEE can be applied to any LLM
by negligible training overhead in advance without affecting the model original
parameters. Extensive experiments show that SpecEE achieves 2.25x and 2.43x
speedup with Llama2-7B on cloud and PC scenarios respectively.

- 使用Eagle的draft model生成topk的token
- 大模型的每一层都经过 topk 的小lm_head，
  - 特征1: 生成Speculative Token Logits
  - 特征2: 再生成Local Probabilities
  - 特征3: 与上一层的Local Probabilities进行对比，得到Probability variation
  - 根据三个特征，经过预测期预测是否要Early Exit
    - 是: 确定Early Exit，计算大lm_head，判断是否与小lm_head的top1一致
      - 一致，推理提前退出
      - 不一致，继续算下一层
    - 否: 继续算下一层
