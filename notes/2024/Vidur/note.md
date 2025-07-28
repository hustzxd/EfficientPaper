# Vidur: A Large-Scale Simulation Framework For LLM Inference

> Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav Gulavani, Ramachandran Ramjee, Alexey Tumanov

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Optimizing the deployment of Large language models (LLMs) is expensive today
since it requires experimentally running an application workload against an LLM
implementation while exploring large configuration space formed by system knobs
such as parallelization strategies, batching techniques, and scheduling
policies. To address this challenge, we present Vidur - a large-scale,
high-fidelity, easily-extensible simulation framework for LLM inference
performance. Vidur models the performance of LLM operators using a combination
of experimental profiling and predictive modeling, and evaluates the end-to-end
inference performance for different workloads by estimating several metrics of
interest such as latency and throughput. We validate the fidelity of Vidur on
several LLMs and show that it estimates inference latency with less than 9%
error across the range. Further, we present Vidur-Search, a configuration
search tool that helps optimize LLM deployment. Vidur-Search uses Vidur to
automatically identify the most cost-effective deployment configuration that
meets application performance constraints. For example, Vidur-Search finds the
best deployment configuration for LLaMA2-70B in one hour on a CPU machine, in
contrast to a deployment-based exploration which would require 42K GPU hours -
costing ~218K dollars. Source code for Vidur is available at
https://github.com/microsoft/vidur.
