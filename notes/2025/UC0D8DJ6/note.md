# Characterizing Communication Patterns in Distributed Large Language Model Inference

> Lang Xu, Kaushik Kandadi Suresh, Quentin Anthony, Nawras Alnaasan, Dhabaleswar K. Panda

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs) built on transformer architectures have
transformed natural language processing, achieving remarkable performance
across diverse applications. While distributed inference frameworks enable
practical deployment of these models, inter-GPU communication creates
significant performance constraints that limit service quality in real-world
systems. This paper investigates communication dynamics in distributed LLM
serving-analyzing how various parallelization approaches coordinate data
exchange between GPU workers during inference. We study dense transformer-based
models as representative examples of contemporary architectures widely used in
operational deployments. Our work combines detailed profiling measurements with
predictive analytical models to characterize communication behavior across
different parallelization configurations. Results show that tensor parallelism
incurs substantial network overhead but delivers superior response times for
brief sequences, pipeline parallelism minimizes data transfer requirements
while increasing total latency, and combined approaches demand careful tuning
to achieve balanced performance. These insights offer practical recommendations
for selecting appropriate parallelization schemes in production LLM services
and identify key opportunities for optimizing inference frameworks and
communication infrastructure.
