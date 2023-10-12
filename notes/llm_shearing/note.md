# Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning

<p align="center">
  <img src="./cover.jpg" width="400" title="kp">
</p>

## structured pruning的两个困难
1. 目前的方法找到的结构是suboptimal的
2. pruned model在不同的任务上保留的能力不同，直接的使用pre-training data进行训练不够高效。

针对以上两个挑战，提出
## Method
1. targeted structured pruning，一种搜索的方法找到更加合适的pruning struture
2. dynamic batch loading，根据domain loss自动调节domain data的比例，从而提升训练效率。