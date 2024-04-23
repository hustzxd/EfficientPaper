# Massive Activations in Large Language Models

<p align="center">
<img src="massive_act.jpg" width="600" title="blank">
</p>

## Abstract

We observe an empirical phenomenon in Large Language Models (LLMs) -- very
few activations exhibit significantly larger values than others (e.g., 100,000
times larger). We call them massive activations. First, we demonstrate the
widespread existence of massive activations across various LLMs and
characterize their locations. Second, we find their values largely stay
constant regardless of the input, and they function as indispensable bias terms
in LLMs. Third, these massive activations lead to the concentration of
attention probabilities to their corresponding tokens, and further, implicit
bias terms in the self-attention output. Last, we also study massive
activations in Vision Transformers.
