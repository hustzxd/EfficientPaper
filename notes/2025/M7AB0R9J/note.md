# GPU-centric Communication Schemes for HPC and ML Applications

> Naveen Namashivayam

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Compute nodes on modern heterogeneous supercomputing systems comprise CPUs,
GPUs, and high-speed network interconnects (NICs). Parallelization is
identified as a technique for effectively utilizing these systems to execute
scalable simulation and deep learning workloads. The resulting inter-process
communication from the distributed execution of these parallel workloads is one
of the key factors contributing to its performance bottleneck. Most programming
models and runtime systems enabling the communication requirements on these
systems support GPU-aware communication schemes that move the GPU-attached
communication buffers in the application directly from the GPU to the NIC
without staging through the host memory. A CPU thread is required to
orchestrate the communication operations even with support for such
GPU-awareness. This survey discusses various available GPU-centric
communication schemes that move the control path of the communication
operations from the CPU to the GPU. This work presents the need for the new
communication schemes, various GPU and NIC capabilities required to implement
the schemes, and the potential use-cases addressed. Based on these discussions,
challenges involved in supporting the exhibited GPU-centric communication
schemes are discussed.
