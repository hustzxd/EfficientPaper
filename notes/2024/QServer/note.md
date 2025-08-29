# QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

![](../../blank.jpg)

## Abstract

Quantization can accelerate large language model (LLM) inference. Going
beyond INT8 quantization, the research community is actively exploring even
lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization
techniques only accelerate low-batch, edge LLM inference, failing to deliver
performance gains in large-batch, cloud-based LLM serving. We uncover a
critical issue: existing INT4 quantization methods suffer from significant
runtime overhead (20-90%) when dequantizing either weights or partial sums on
GPUs. To address this challenge, we introduce QoQ, a W4A8KV4 quantization
algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands
for quattuor-octo-quattuor, which represents 4-8-4 in Latin. QoQ is implemented
by the QServe inference library that achieves measured speedup. The key insight
driving QServe is that the efficiency of LLM serving on GPUs is critically
influenced by operations on low-throughput CUDA cores. Building upon this
insight, in QoQ algorithm, we introduce progressive quantization that can allow
low dequantization overhead in W4A8 GEMM. Additionally, we develop
SmoothAttention to effectively mitigate the accuracy degradation incurred by
4-bit KV quantization. In the QServe system, we perform compute-aware weight
reordering and take advantage of register-level parallelism to reduce
dequantization latency. We also make fused attention memory-bound, harnessing
the performance gain brought by KV4 quantization. As a result, QServe improves
the maximum achievable serving throughput of Llama-3-8B by 1.2x on A100, 1.4x
on L40S; and Qwen1.5-72B by 2.4x on A100, 3.5x on L40S, compared to
TensorRT-LLM. Remarkably, QServe on L40S GPU can achieve even higher throughput
than TensorRT-LLM on A100. Thus, QServe effectively reduces the dollar cost of
LLM serving by 3x. Code is available at https://github.com/mit-han-lab/qserve.
