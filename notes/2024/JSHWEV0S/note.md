# Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption

<p align="center">
<img src="overview.png" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs), epitomized by ChatGPT' s release in late 2022,
have revolutionized various industries with their advanced language
comprehension. However, their efficiency is challenged by the Transformer
architecture' s struggle with handling long texts. KV-Cache has emerged as a
pivotal solution to this issue, converting the time complexity of token
generation from quadratic to linear, albeit with increased GPU memory overhead
proportional to conversation length. With the development of the LLM community
and academia, various KV-Cache compression methods have been proposed. In this
review, we dissect the various properties of KV-Cache and elaborate on various
methods currently used to optimize the KV-Cache space usage of LLMs. These
methods span the pre-training phase, deployment phase, and inference phase, and
we summarize the commonalities and differences among these methods.
Additionally, we list some metrics for evaluating the long-text capabilities of
large language models, from both efficiency and capability perspectives. Our
review thus sheds light on the evolving landscape of LLM optimization, offering
insights into future advancements in this dynamic field.
