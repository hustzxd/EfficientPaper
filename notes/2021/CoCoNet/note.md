# Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads

> Abhinav Jangda, Jun Huang, Guodong Liu, Amir Hossein Nodehi Sabet, Saeed Maleki, Youshan Miao, Madanlal Musuvathi, Todd Mytkowicz, Olli Sarikivi

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Recent trend towards increasing large machine learning models require both
training and inference tasks to be distributed. Considering the huge cost of
training these models, it is imperative to unlock optimizations in computation
and communication to obtain best performance. However, current logical
separation between computation and communication kernels in deep learning
frameworks misses the optimization opportunities across such barrier. Breaking
this abstraction with a holistic consideration can provide many optimizations
to provide performance improvements in distributed workloads. Manually applying
these optimizations needs modifications in underlying computation and
communication libraries for each scenario, which is time consuming and
error-prone.
  Therefore, we present CoCoNeT, with a DSL to express a program with both
computation and communication. CoCoNeT contains several machine learning aware
transformations to optimize a program and a compiler to generate high
performance kernels. Providing both computation and communication as first
class constructs allows users to work on a high-level abstraction and apply
powerful optimizations, such as fusion or overlapping of communication and
computation. CoCoNeT enables us to optimize data-, model-and pipeline-parallel
workloads in large language models with only a few lines of code. Experiments
show CoCoNeT significantly outperforms state-of-the-art distributed machine
learning implementations.
