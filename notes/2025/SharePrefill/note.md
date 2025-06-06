# Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing

<p align="center">
<img src="fig3.png" width="600" title="blank">
</p>

## Abstract

Sparse attention methods exploit the inherent sparsity in attention to speed
up the prefilling phase of long-context inference, mitigating the quadratic
complexity of full attention computation. While existing sparse attention
methods rely on predefined patterns or inaccurate estimations to approximate
attention behavior, they often fail to fully capture the true dynamics of
attention, resulting in reduced efficiency and compromised accuracy. Instead,
we propose a highly accurate sparse attention mechanism that shares similar yet
precise attention patterns across heads, enabling a more realistic capture of
the dynamic behavior of attention. Our approach is grounded in two key
observations: (1) attention patterns demonstrate strong inter-head similarity,
and (2) this similarity remains remarkably consistent across diverse inputs. By
strategically sharing computed accurate patterns across attention heads, our
method effectively captures actual patterns while requiring full attention
computation for only a small subset of heads. Comprehensive evaluations
demonstrate that our approach achieves superior or comparable speedup relative
to state-of-the-art methods while delivering the best overall accuracy.

根据统计，将head 聚类，同一个cluster内部共享block sparse mask，计算时，cluster中第一个head按照dense计算，得到输入后计算sparse mask，此后这个cluster中的其他head，便可以根据mask进行sparse 计算。