# SlimLLM: Accurate Structured Pruning for Large Language Models

> Jialong Guo, Xinghao Chen, Yehui Tang, Yunhe Wang

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Large language models(LLMs) have garnered significant attention and
demonstrated impressive capabilities in a wide range of applications. However,
due to their enormous computational costs, the deployment and application of
LLMs are often severely limited. To address this issue, structured pruning is
an effective solution to compress the parameters of LLMs. Determining the
importance of each sub-module in LLMs and minimizing performance loss are
critical issues that need to be carefully addressed in structured pruning. In
this paper, we propose an effective and fast structured pruning method named
SlimLLM for large language models. For channel and attention head pruning, we
evaluate the importance based on the entire channel or head, rather than merely
aggregating the importance of individual elements within a sub-module. This
approach enables a more holistic consideration of the interdependence among
elements within the sub-module. In addition, we design a simple linear
regression strategy for the output matrix to quickly recover performance. We
also propose layer-based importance ratio to determine the pruning ratio for
each layer. Based on the LLaMA benchmark results, our SlimLLM outperforms other
methods and achieves state-of-the-art performance.
