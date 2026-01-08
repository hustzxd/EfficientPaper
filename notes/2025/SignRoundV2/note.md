# SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs

> Wenhua Cheng, Weiwei Zhang, Heng Guo, Haihao Shen

![111](../../blank.jpg)

## Abstract

Extreme low-bit quantization is critical for efficiently deploying Large Language Models (LLMs), yet it often leads to severe performance degradation at 2-bits and even 4-bits (e.g., MXFP4). We present SignRoundV2, a post-training quantization framework that is highly effective even without mixed-precision. SignRoundV2 introduces (1) a fast sensitivity metric that combines gradient information with quantization-induced deviations to guide layer-wise bit allocation, and (2) a lightweight pre-tuning search for quantization scales to improve extremely low-bit quantization. These components allow SignRoundV2 to close the gap with full-precision models. Extensive experiments indicate that our method sustains competitive accuracy for LLMs, achieving production-grade performance with about 1 percent variance at 4-5 bits and strong results even at 2 bits. The implementation is available at https://github.com/intel/auto-round.
