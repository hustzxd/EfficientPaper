# EvolKV: Evolutionary KV Cache Compression for LLM Inference

> Bohan Yu, Yekun Chai

![111](fig1.png)

## Abstract

Existing key-value (KV) cache compression methods typically rely on
heuristics, such as uniform cache allocation across layers or static eviction
policies, however, they ignore the critical interplays among layer-specific
feature patterns and task performance, which can lead to degraded
generalization. In this paper, we propose EvolKV, an adaptive framework for
layer-wise, task-driven KV cache compression that jointly optimizes the memory
efficiency and task performance. By reformulating cache allocation as a
multi-objective optimization problem, EvolKV leverages evolutionary search to
dynamically configure layer budgets while directly maximizing downstream
performance. Extensive experiments on 11 tasks demonstrate that our approach
outperforms all baseline methods across a wide range of KV cache budgets on
long-context tasks and surpasses heuristic baselines by up to 7 percentage
points on GSM8K. Notably, EvolKV achieves superior performance over the full KV
cache setting on code completion while utilizing only 1.5% of the original
budget, suggesting the untapped potential in learned compression strategies for
KV cache budget allocation.

- 优化目标跟task绑定，选取30个samples
- 使用进化算法分配kv budget
- 任务不同，分配方式也不同