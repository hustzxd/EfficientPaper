# QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization

> Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Inference accounts for the majority of latency and energy consumption in
large language model (LLM) deployments, often exceeding 90% of total cost.
While training-time efficiency has seen extensive progress, runtime
optimization remains a key bottleneck, particularly under autoregressive
decoding. Existing approaches -- such as pruning, quantization, early exits,
and speculative decoding -- often require retraining, architectural changes, or
disrupt decoding compatibility. We introduce QuickSilver, a modular,
token-level framework that enables semantic adaptivity at inference time
without altering model weights or structure. QuickSilver integrates four
synergistic mechanisms:
  (i) Dynamic Token Halting, which halts computation for tokens with converged
representations; (ii) KV Cache Skipping, which selectively suppresses memory
writes to reduce attention overhead; and (iii) Contextual Token Fusion, which
collapses redundant tokens into shared paths to shrink sequence length.
  Unlike speculative decoding or MoE routing, QuickSilver operates entirely on
frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and
Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP
reduction with negligible perplexity degradation (<=0.2).
