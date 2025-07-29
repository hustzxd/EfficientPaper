# Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need

> Michael Davies, Neal Crago, Karthikeyan Sankaralingam, Christos Kozyrakis

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

This paper presents a limit study of transformer-based large language model
(LLM) inference, focusing on the fundamental performance bottlenecks imposed by
memory bandwidth, memory capacity, and synchronization overhead in distributed
inference systems. We develop a hardware-agnostic performance model that
abstracts away implementation details, enabling the analysis of a wide range of
current and near-future hardware technologies. Our analysis spans from current
HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems
based on advanced HBM4 and advanced 3D-stacked DRAM technology. It also covers
SRAM-based designs and scaling techniques from distributed clusters with
varying numbers of chips to wafer-scale integration. Our key findings for
auto-regressive decoding are: i) serving LLMs requires 100s of GB per server to
serve a model instance; ii) high memory bandwidth is critical for high per-user
throughput; iii) exposed synchronization latencies to achieve collective
communication must be around 1us else they make the memory bandwidth
ineffective; iv) DRAM-based designs have a fundamental advantage in terms of
system-level efficiency as measured in throughput per cost or watt; and v)
hardware designs can easily reach 2000+ user token/sec but getting to 10,000+
tokens/sec will need smaller models, smaller context, or other forms of
algorithmic advances. This study provides valuable insights into the
fundamental performance limits of LLM inference, highlighting the potential
benefits of future hardware advancements and guiding the optimization of LLM
deployment strategies.
