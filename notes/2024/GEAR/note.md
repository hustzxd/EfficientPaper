# GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM

> Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao

![111](overview.png)

## Abstract

Key-value (KV) caching has become the de-facto to accelerate generation speed
for large language models (LLMs) inference. However, the growing cache demand
with increasing sequence length has transformed LLM inference to be a memory
bound problem, significantly constraining the system throughput. Existing
methods rely on dropping unimportant tokens or quantizing all entries
uniformly. Such methods, however, often incur high approximation errors to
represent the compressed matrices. The autoregressive decoding process further
compounds the error of each step, resulting in critical deviation in model
generation and deterioration of performance. To tackle this challenge, we
propose GEAR, an efficient KV cache compression framework that achieves
near-lossless high-ratio compression. GEAR first applies quantization to
majority of entries of similar magnitudes to ultra-low precision. It then
employs a low rank matrix to approximate the quantization error, and a sparse
matrix to remedy individual errors from outlier entries. By adeptly integrating
three techniques, GEAR is able to fully exploit their synergistic potentials.
Our experiments demonstrate that compared to alternatives, GEAR achieves
near-lossless 4-bit KV cache compression with up to 2.38x throughput
improvement, while reducing peak-memory size up to 2.29x. Our code is publicly
available at https://github.com/HaoKang-Timmy/GEAR.

- quant + 低秩分解