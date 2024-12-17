# XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

The applications of LLM Agents are becoming increasingly complex and diverse,
leading to a high demand for structured outputs that can be parsed into code,
structured function calls, and embodied agent commands. These developments
bring significant demands for structured generation in LLM inference.
Context-free grammar is a flexible approach to enable structured generation via
constrained decoding. However, executing context-free grammar requires going
through several stack states over all tokens in vocabulary during runtime,
bringing non-negligible overhead for structured generation. In this paper, we
propose XGrammar, a flexible and efficient structure generation engine for
large language models. XGrammar accelerates context-free grammar execution by
dividing the vocabulary into context-independent tokens that can be prechecked
and context-dependent tokens that need to be interpreted during runtime. We
further build transformations to expand the grammar context and reduce the
number of context-independent tokens. Additionally, we build an efficient
persistent stack to accelerate the context-dependent token checks. Finally, we
co-design the grammar engine with LLM inference engine to overlap grammar
computation with GPU executions. Evaluation results show that XGrammar can
achieve up to 100x speedup over existing solutions. Combined with an LLM
inference engine, it can generate near-zero overhead structure generation in
end-to-end low-LLM serving.
