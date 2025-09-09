# MoPEQ: Mixture of Mixed Precision Quantized Experts

> Krishna Teja Chitty-Venkata, Jie Ye, Murali Emani

![111](fig1.png)

## Abstract

Large Language and Vision Models using a Mixture-of-Experts (MoE)
architecture pose significant challenges for deployment due to their
computational and memory demands. Mixed Precision Quantization assigns
different precisions to different layers of an LLM/VLM based on layer
sensitivity and importance within the model. In this work, we propose a Post
Training Quantization algorithm, MoPEQ, that assigns optimal bit width to each
expert. Our method balances accuracy and model size by analyzing each expert's
sensitivity using Hessian trace approximation instead of relying on the
activation frequency of the expert. This per-expert granularity approach
clusters similar experts to maintain model performance while reducing memory
requirements. The experimental results on VLMEvalKit benchmark datasets using
State-of-the-art VLMs Deepseek-VL2 -tiny, -small, -base, and MolmoE models
demonstrate that our mixed precision quantized MoEs achieve competitive
accuracy with substantial improvements in memory footprint compared to
uniform-precision baseline methods. We perform a comprehensive study to analyze
the impact of expert activation frequency and sensitivity using Hessian trace
approximation at both layer-wise and model-wide expert precision allocation of
2, 3, and 4 bits to provide a thorough understanding of mixed precision
quantization of VLM-MoEs.
