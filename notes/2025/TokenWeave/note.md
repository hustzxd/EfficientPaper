# TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference

> Raja Gond, Nipun Kwatra, Ramachandran Ramjee

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Distributed inference of large language models (LLMs) can introduce overheads
of up to 20% even over GPUs connected via high-speed interconnects such as
NVLINK. Multiple techniques have been proposed to mitigate these overheads by
decomposing computations into finer-grained tasks and overlapping communication
with sub-tasks as they complete. However, fine-grained decomposition of a large
computation into many smaller computations on GPUs results in overheads.
Further, the communication itself uses many streaming multiprocessors (SMs),
adding to the overhead.
  We present TokenWeave to address these challenges. TokenWeave proposes a
Token-Splitting technique that divides the tokens in the inference batch into
two approximately equal subsets in a wave-aware manner. The computation of one
subset is then overlapped with the communication of the other. In addition,
TokenWeave optimizes the order of the layer normalization computation with
respect to communication operations and implements a novel fused
AllReduce-RMSNorm kernel carefully leveraging Multimem instruction support
available on NVIDIA Hopper GPUs. These optimizations allow TokenWeave to
perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel
enables the memory bound RMSNorm to be overlapped with the other batch's
computation, providing additional gains. Our evaluations demonstrate up to 29%
latency gains and up to 26% throughput gains across multiple models and
workloads. In several settings, TokenWeave results in better performance
compared to an equivalent model with all communication removed.
