# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

![](fig1.png)

## Abstract

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and
DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement
learning (RL) without supervised fine-tuning (SFT) as a preliminary step,
demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero
naturally emerges with numerous powerful and intriguing reasoning behaviors.
However, it encounters challenges such as poor readability, and language
mixing. To address these issues and further enhance reasoning performance, we
introduce DeepSeek-R1, which incorporates multi-stage training and cold-start
data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217
on reasoning tasks. To support the research community, we open-source
DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B,
70B) distilled from DeepSeek-R1 based on Qwen and Llama.
