# MoBA: Mixture of Block Attention for Long-Context LLMs

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Scaling the effective context length is essential for advancing large
language models (LLMs) toward artificial general intelligence (AGI). However,
the quadratic increase in computational complexity inherent in traditional
attention mechanisms presents a prohibitive overhead. Existing approaches
either impose strongly biased structures, such as sink or window attention
which are task-specific, or radically modify the attention mechanism into
linear approximations, whose performance in complex reasoning tasks remains
inadequately explored.
  In this work, we propose a solution that adheres to the ``less structure''
principle, allowing the model to determine where to attend autonomously, rather
than introducing predefined biases. We introduce Mixture of Block Attention
(MoBA), an innovative approach that applies the principles of Mixture of
Experts (MoE) to the attention mechanism. This novel architecture demonstrates
superior performance on long-context tasks while offering a key advantage: the
ability to seamlessly transition between full and sparse attention, enhancing
efficiency without the risk of compromising performance. MoBA has already been
deployed to support Kimi's long-context requests and demonstrates significant
advancements in efficient attention computation for LLMs. Our code is available
at https://github.com/MoonshotAI/MoBA.
