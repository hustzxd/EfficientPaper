# GPU-Accelerated INT8 Quantization for KV Cache Compression in Large Language Models

> Maanas Taneja, Purab Shingvi

![111](../../blank.jpg)

## Abstract

The key-value (KV) cache in large language models presents a significant memory bottleneck during inference, growing linearly with sequence length and often exceeding the memory footprint of model weights themselves. We implement and evaluate GPU-accelerated INT8 quantization for KV cache compression, achieving 4$\times$ memory reduction with minimal accuracy degradation. We develop four CUDA kernel variants -- naive, tiled, coarsened, and vectorized -- and benchmark them across realistic workload sizes up to 1 billion elements. Our vectorized kernel achieves up to 1,694$\times$ speedup over CPU baselines while maintaining reconstruction error below 0.004 and attention score error below 0.1 even for 8K-dimensional heads. These results demonstrate that INT8 quantization provides a practical approach for reducing memory pressure in LLM inference with negligible computational overhead (6--58ms) and minimal impact on downstream model behavior
