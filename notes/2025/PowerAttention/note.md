# PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs) face efficiency bottlenecks due to the quadratic
complexity of the attention mechanism when processing long contexts. Sparse
attention methods offer a promising solution, but existing approaches often
suffer from incomplete effective context and/or require complex implementation
of pipeline. We present a comprehensive analysis of sparse attention for
autoregressive LLMs from the respective of receptive field, recognize the
suboptimal nature of existing methods for expanding the receptive field, and
introduce PowerAttention, a novel sparse attention design that facilitates
effective and complete context extension through the theoretical analysis.
PowerAttention achieves exponential receptive field growth in $d$-layer LLMs,
allowing each output token to attend to $2^d$ tokens, ensuring completeness and
continuity of the receptive field. Experiments demonstrate that PowerAttention
outperforms existing static sparse attention methods by $5\sim 40\%$,
especially on tasks demanding long-range dependencies like Passkey Retrieval
and RULER, while maintaining a comparable time complexity to sliding window
attention. Efficiency evaluations further highlight PowerAttention's superior
speedup in both prefilling and decoding phases compared with dynamic sparse
attentions and full attention ($3.0\times$ faster on 128K context), making it a
highly effective and user-friendly solution for processing long sequences in
LLMs.
