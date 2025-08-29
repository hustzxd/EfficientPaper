# Instruction-Following Pruning for Large Language Models

![](fig1.png)

## Abstract

With the rapid scaling of large language models (LLMs), structured pruning
has become a widely used technique to learn efficient, smaller models from
larger ones, delivering superior performance compared to training similarly
sized models from scratch. In this paper, we move beyond the traditional static
pruning approach of determining a fixed pruning mask for a model, and propose a
dynamic approach to structured pruning. In our method, the pruning mask is
input-dependent and adapts dynamically based on the information described in a
user instruction. Our approach, termed "instruction-following pruning",
introduces a sparse mask predictor that takes the user instruction as input and
dynamically selects the most relevant model parameters for the given task. To
identify and activate effective parameters, we jointly optimize the sparse mask
predictor and the LLM, leveraging both instruction-following data and the
pre-training corpus. Experimental results demonstrate the effectiveness of our
approach on a wide range of evaluation benchmarks. For example, our 3B
activated model improves over the 3B dense model by 5-8 points of absolute
margin on domains such as math and coding, and rivals the performance of a 9B
model.

引入了Sparsity Predictor决定MLP的部分channel被稀疏。这个Predictor仅和prompt有关，所以介于 input-dependent pruning 与 static pruning 之间。
这个方法依赖训练，需要训练Predictor，比如”translate English text to French“作为prompt，多轮对话时，只使用第一轮human的message作为prompt。
在推理时，仅需要根据prompt输入到Predictor得到MLP的稀疏位置，得到sub model，使用sub model进行推理，便可以提高性能。

新的挑战，不同task就不能进行batching了，因为他们激活的位置是不同的。也就要求提前对应用进行分类，针对每个应用得到稀疏后的sub model，从而可以加速每个task。