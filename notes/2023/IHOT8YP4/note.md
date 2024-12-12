# Efficient Guided Generation for Large Language Models

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

In this article we show how the problem of neural text generation can be
constructively reformulated in terms of transitions between the states of a
finite-state machine. This framework leads to an efficient approach to guiding
text generation with regular expressions and context-free grammars by allowing
the construction of an index over a language model's vocabulary. The approach
is model agnostic, allows one to enforce domain-specific knowledge and
constraints, and enables the construction of reliable interfaces by
guaranteeing the structure of the generated text. It adds little overhead to
the token sequence generation process and significantly outperforms existing
solutions. An implementation is provided in the open source Python library
Outlines
