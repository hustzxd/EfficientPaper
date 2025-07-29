# Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications

> Seonho Lee, Jihwan Oh, Junkyum Kim, Seokjin Go, Jongse Park, Divya Mahajan

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

This paper provides an in-depth characterization of GPU-accelerated systems,
to understand the interplay between overlapping computation and communication
which is commonly employed in distributed training settings. Due to the large
size of models, distributing them across multiple devices is required.
Overlapping strategies, which enable concurrent computation and communication,
are critical for mitigating communication bottlenecks and maximizing GPU
utilization. However, the current consensus is that we should always and
aggressively overlap compute and communication to mitigate the overhead of
distribution. By systematically evaluating state-of-the-art GPUs, this study
investigates the impact of hardware features such as numeric precision,
specialized cores, and power capping on distributed training workloads.
Comprehensive experiments and studies showcase the effects of overlapping
strategies on performance and power consumption across varying scenarios. We
observe that overlapping computation and communication can result in an average
computational slowdown of 18.9%, with a maximum of 40.0% slowdown. This
slowdown is in comparison to the scenario when no communication was happening
with the compute. We consider this an ideal execution scenario, where the
communication in parallel has not impact on the compute time. However,
performing computation and communication sequentially is, on average, 10.2%
slower than overlapped execution, with a maximum slowdown of 26.6%. We further
observe, while specialized datapath and optimized numeric precision mitigate
certain slowdowns, overlapping execution can lead to resource contention and
also increase power consumption under specific configurations. The analysis
also uncovers trade-offs introduced by power and frequency capping, emphasizing
the importance of balanced strategies to optimize energy efficiency and
training throughput.
