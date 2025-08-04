# APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving

> Yi-Chien Lin, Woosuk Kwon, Ronald Pineda, Fanny Nina Paravecino

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Efficiently serving Large Language Models (LLMs) requires selecting an
optimal parallel execution plan, balancing computation, memory, and
communication overhead. However, determining the best strategy is challenging
due to varying parallelism techniques (data, pipeline, tensor) and workload
characteristics (e.g., compute-intensive tasks with long prompts vs.
memory-intensive tasks with long generation). We propose APEX, an LLM serving
system simulator that efficiently identifies optimal parallel execution plans
by considering key factors of LLM serving systems, such as memory usage,
batching behavior, etc. APEX performs dynamism-aware simulation to model
iteration-level batching, and leverages LLMs' repetitive structure to reduce
design space, scaling efficiently to trillion-scale models. APEX abstracts the
key components of LLM serving systems, including the model, batching module,
quantization formats, and device clusters, enabling the simulator to be general
and extensible. Simulating on a CPU, APEX evaluates execution plans for various
device clusters, covering diverse LLMs and workloads. APEX finds plans up to
3.37x faster than heuristics, and also plans that reduce energy consumption by
up to 45% compared to latency-optimal plans. APEX performs comprehensive
evaluations, reporting key system metrics like time per output token and time
to first token, which can help service providers meet SLOs. APEX identifies an
optimal plan within 15 minutes on a CPU, making it 71x faster and 1234x more
cost-effective than cloud-based GPU deployment. APEX can be accessed at
https://github.com/microsoft/apex_plus
