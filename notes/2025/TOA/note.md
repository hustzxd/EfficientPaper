# Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning

> Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian

![111](fig1.png)

## Abstract

Large language models (LLMs) face persistent challenges when handling
long-context tasks, most notably the lost in the middle issue, where
information located in the middle of a long input tends to be underutilized.
Some existing methods that reduce input have the risk of discarding key
information, while others that extend context windows often lead to attention
dispersion. To address these limitations, we propose Tree of Agents (TOA), a
multi-agent reasoning framework that segments the input into chunks processed
by independent agents. Each agent generates its local cognition, then agents
dynamically exchange information for collaborative reasoning along
tree-structured paths. TOA enables agents to probe different reasoning orders
for multi-perspective understanding, effectively mitigating position bias and
reducing hallucinations. To improve processing efficiency, we incorporate
prefix-hash caching and adaptive pruning strategies, achieving significant
performance improvements with comparable API overhead. Experiments show that
TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple
baselines and demonstrates comparable performance to the latest and much larger
commercial models, such as Gemini1.5-pro, on various long-context tasks. Code
is available at https://github.com/Aireduce952/Tree-of-Agents.
