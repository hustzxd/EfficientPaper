# MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization

> Zongwu Wang, Peng Xu, Fangxin Liu, Yiwei Hu, Qingxiao Sun, Gezi Li, Cheng Li, Xuan Wang, Li Jiang, Haibing Guan

![111](fig4.png)

## Abstract

Large language models (LLMs) are increasingly utilized for complex tasks
requiring longer context lengths, with some models supporting up to 128K or 1M
tokens. This trend, however, presents significant challenges in inference speed
and memory management. Quantization emerges as a promising approach to address
the widening gap between LLM size and memory capacity. However, traditional
quantization schemes often yield suboptimal compression results for KV caches
due to two key factors: i) On-the-fly quantization and de-quantization, causing
significant performance overhead; ii) Prevalence of outliers in KV values,
challenging low-bitwidth uniform quantization. To this end, we propose MILLION,
a novel quantization framework achieving low-bitwidth KV cache through product
quantization. First, we conduct a thorough analysis of KV cache distribution,
revealing the limitations of existing quantization schemes. Second, we
introduce a non-uniform quantization algorithm based on product quantization,
which efficiently compresses data while preserving accuracy. Third, we develop
a high-performance GPU inference framework with efficient attention kernel and
pipeline design for MILLION that leverages sparse computation and asynchronous
quantization, significantly enhancing inference speed. Comprehensive evaluation
results demonstrate that MILLION can achieve 4 bits quantization with trivial
perplexity and accuracy loss, and achieve 2.09x end-to-end performance gains at
32K context length. Code is released at https://github.com/ZongwuWang/MILLION.
