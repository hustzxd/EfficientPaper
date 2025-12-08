# EfficientPaper
Pruning, Quantization and efficient-inference/training paper list.

## Table of Contents
- [EfficientPaper](#efficientpaper)
  - [Getting Started](#getting-started)
  - [Paper List](#paper-list)
    - [keyword](cls_keyword.md)
    - [year](cls_year.md)
    - [publication](cls_publication.md)
    - [institution](cls_institution.md)
    - [author](cls_author.md)
  - [References](#references)



## Getting Started
```bash
git clone https://github.com/hustzxd/EfficientPaper
pip install protobuf==5.27.2 pandas arxiv 
```
1. Add paper information by `./add_paper_info.sh`
2. Run `./refresh_readme.sh`

<details><summary><b>efficient_paper.prototxt</b></summary>	
<p>

```
paper {
  title: "EfficientPaper: manage your research papers in an efficient way."
  abbr: "EfficientPaper"
  url: "https://github.com/hustzxd/EfficientPaper"
  authors: "hustzxd"
}
pub {
  where: "GitHub"
  year: 2023
}
code {
  type: "Pytorch"
  url: "https://github.com/hustzxd/EfficientPaper"
}
note {
  url: "EfficientPaper.md"
}
keyword {
  words: efficient_paper
}
```

</p>
</details>

<p align="center">
<img src="notes//conference_timeline.png" width="800" title="blank">
</p>


## 招聘

如果您对论文涉及到的研究内容感兴趣，同时有求职意向（[实习生/校招/社招](https://m.zhipin.com/gongsi/job/dc8e21b748a34c331HZz3Nu-GFU~.html?ka=m_seo_companys_all_jobs_boss)），可以发送简历到zhaoxiandong27@gmail.com，欢迎沟通交流。


## Paper List

<details open><summary>

### year
</summary> 
<p>

<details open><summary><b>2026</b></summary> 
<p>

1. [FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation](http://arxiv.org/abs/2504.19519v1) [![Publish](https://img.shields.io/badge/2026-EuroSys-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/infinigence/FlashOverlap)](https://github.com/infinigence/FlashOverlap) 
</p>
</details>
<details open><summary><b>2025</b></summary> 
<p>

1. [AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference](http://arxiv.org/abs/2501.02336v1) [![Publish](https://img.shields.io/badge/2025-AAAI-FF4500)] [![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/AdaSkip)](https://github.com/ASISys/AdaSkip) 
2. [Pruning Large Language Models with Semi-Structural Adaptive Sparse Training](http://arxiv.org/abs/2407.20584v3) [![Publish](https://img.shields.io/badge/2025-AAAI-FF4500)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/Adaptive-Sparse-Trainer)](https://github.com/thu-ml/Adaptive-Sparse-Trainer) 
3. [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](http://arxiv.org/abs/2406.03482v2) [![Publish](https://img.shields.io/badge/2025-AAAI-FF4500)] [![GitHub Repo stars](https://img.shields.io/github/stars/amirzandieh/QJL)](https://github.com/amirzandieh/QJL) 
4. [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](http://arxiv.org/abs/2502.11089v1) [![Publish](https://img.shields.io/badge/2025-ACL-4169E1)]  
5. [Training LLMs with MXFP4](http://arxiv.org/abs/2502.20586v2) [![Publish](https://img.shields.io/badge/2025-AISTATS-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/amazon-science/mxfp4-llm)](https://github.com/amazon-science/mxfp4-llm) 
6. [COMET: Towards Partical W4A4KV4 LLMs Serving](http://arxiv.org/abs/2410.12168v1) [![Publish](https://img.shields.io/badge/2025-ASPLOS-9370DB)]  
7. [POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference](http://arxiv.org/abs/2410.18038v2) [![Publish](https://img.shields.io/badge/2025-ASPLOS-9370DB)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/vattention)](https://github.com/microsoft/vattention) 
8. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](http://arxiv.org/abs/2405.04437v3) [![Publish](https://img.shields.io/badge/2025-ASPLOS-9370DB)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/vattention)](https://github.com/microsoft/vattention) 
9. [BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity](http://arxiv.org/abs/2507.08771v1) [![Publish](https://img.shields.io/badge/2025-COLM-6495ED)] [![GitHub Repo stars](https://img.shields.io/github/stars/thunlp/BlockFFN)](https://github.com/thunlp/BlockFFN) 
10. [KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs](http://arxiv.org/abs/2508.04257v1) [![Publish](https://img.shields.io/badge/2025-COLM-6495ED)]  
11. [Enhancing One-shot Pruned Pre-trained Language Models through Sparse-Dense-Sparse Mechanism](http://arxiv.org/abs/2408.10473v1) [![Publish](https://img.shields.io/badge/2025-Coling-1E90FF)]  
12. [MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization](http://arxiv.org/abs/2504.03661v2) [![Publish](https://img.shields.io/badge/2025-DAC-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/ZongwuWang/MILLION)](https://github.com/ZongwuWang/MILLION) 
13. [UNComp: Can Matrix Entropy Uncover Sparsity? -- A Compressor Design from an Uncertainty-Aware Perspective](http://arxiv.org/abs/2410.03090v2) [![Publish](https://img.shields.io/badge/2025-EMNLP-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/menik1126/UNComp)](https://github.com/menik1126/UNComp) 
14. [Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning](http://arxiv.org/abs/2509.06436v1) [![Publish](https://img.shields.io/badge/2025-EMNLP_Findings-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/Aireduce952/Tree-of-Agents)](https://github.com/Aireduce952/Tree-of-Agents) 
15. [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](http://arxiv.org/abs/2405.16444v3) [![Publish](https://img.shields.io/badge/2025-EuroSys-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/LMCache/LMCache)](https://github.com/LMCache/LMCache) 
16. [SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs](https://dl.acm.org/doi/10.1145/3689031.3717481) [![Publish](https://img.shields.io/badge/2025-EuroSys-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/xxyux/SpInfer)](https://github.com/xxyux/SpInfer) 
17. [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) [![Publish](https://img.shields.io/badge/2025-Github-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V3.2-Exp)](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 
18. [FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference](http://arxiv.org/abs/2502.20766v1) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/FlexPrefill)](https://github.com/bytedance/FlexPrefill) 
19. [Forgetting Transformer: Softmax Attention with a Forget Gate](http://arxiv.org/abs/2503.02130v2) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/zhixuan-lin/forgetting-transformer)](https://github.com/zhixuan-lin/forgetting-transformer) 
20. [R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](http://arxiv.org/abs/2504.19449v1) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/R-Sparse)](https://github.com/VITA-Group/R-Sparse) 
21. [ReAttention: Training-Free Infinite Context with Finite Attention Scope](http://arxiv.org/abs/2407.15176v3) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenMOSS/ReAttention)](https://github.com/OpenMOSS/ReAttention) 
22. [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](http://arxiv.org/abs/2410.20672v3) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)]  
23. [TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention](http://arxiv.org/abs/2410.05076v1) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/DerrickYLJ/TidalDecode)](https://github.com/DerrickYLJ/TidalDecode) 
24. [Training-Free Activation Sparsity in Large Language Models](http://arxiv.org/abs/2408.14690v1) [![Publish](https://img.shields.io/badge/2025-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/FasterDecoding/TEAL)](https://github.com/FasterDecoding/TEAL) 
25. [AdaSplash: Adaptive Sparse Flash Attention](http://arxiv.org/abs/2502.12082) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/deep-spin/adasplash)](https://github.com/deep-spin/adasplash) 
26. [BaWA: Automatic Optimizing Pruning Metric for Large Language Models with Balanced Weight and Activation](https://openreview.net/forum?id=YrCvW1Hx7g) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
27. [CateKV: On Sequential Consistency for Long-Context LLM Inference Acceleration](https://openreview.net/forum?id=u7dlwgKstN) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
28. [Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity](http://arxiv.org/abs/2412.02252v1) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
29. [HashAttention: Semantic Sparsity for Faster Inference](https://openreview.net/forum?id=Em2oaXd8Dc) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/xAlg-ai/HashAttention-1.0)](https://github.com/xAlg-ai/HashAttention-1.0) 
30. [La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](http://arxiv.org/abs/2507.01299v1) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
31. [MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](https://openreview.net/forum?id=me6PfbATWM) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
32. [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://openreview.net/forum?id=oa7MYAO6h6) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/ShadowKV)](https://github.com/bytedance/ShadowKV) 
33. [SlimLLM: Accurate Structured Pruning for Large Language Models](http://arxiv.org/abs/2505.22689v1) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)]  
34. [SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](https://openreview.net/forum?id=74c3Wwk8Tc) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SpargeAttn)](https://github.com/thu-ml/SpargeAttn) 
35. [Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity](http://arxiv.org/abs/2502.01776v2) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/svg-project/Sparse-VideoGen)](https://github.com/svg-project/Sparse-VideoGen) 
36. [Sparsing Law: Towards Large Language Models with Greater Activation Sparsity](https://openreview.net/forum?id=SBUc5wirM8) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/thunlp/SparsingLaw)](https://github.com/thunlp/SparsingLaw) 
37. [Star Attention: Efficient LLM Inference over Long Sequences](https://openreview.net/forum?id=QY7Au9nZwp) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/Star-Attention)](https://github.com/NVIDIA/Star-Attention) 
38. [XAttention: Block Sparse Attention with Antidiagonal Scoring](http://arxiv.org/abs/2503.16428v1) [![Publish](https://img.shields.io/badge/2025-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/x-attention)](https://github.com/mit-han-lab/x-attention) 
39. [TorchAO: PyTorch-Native Training-to-Serving Model Optimization](http://arxiv.org/abs/2507.16099v1) [![Publish](https://img.shields.io/badge/2025-ICML_Workshop-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/pytorch/ao)](https://github.com/pytorch/ao) 
40. [AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs](https://dl.acm.org/doi/10.1145/3695053.3731064) [![Publish](https://img.shields.io/badge/2025-ISCA-9932CC)]  
41. [SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting](http://arxiv.org/abs/2504.08850v1) [![Publish](https://img.shields.io/badge/2025-ISCA-9932CC)] [![GitHub Repo stars](https://img.shields.io/github/stars/infinigence/SpecEE)](https://github.com/infinigence/SpecEE) 
42. [Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models](http://arxiv.org/abs/2508.01261v1) [![Publish](https://img.shields.io/badge/2025-KDD_Workshop-green)]  
43. [Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving](http://arxiv.org/abs/2503.24000v1) [![Publish](https://img.shields.io/badge/2025-MLSys-DDA0DD)] [![GitHub Repo stars](https://img.shields.io/github/stars/LLMkvsys/rethink-kv-compression)](https://github.com/LLMkvsys/rethink-kv-compression) 
44. [Delta Attention: Fast and Accurate Sparse Attention Inference by Delta Correction](http://arxiv.org/abs/2505.11254v1) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)]  
45. [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](http://arxiv.org/abs/2505.06708v1) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/qiuzh20/gated_attention)](https://github.com/qiuzh20/gated_attention) 
46. [MoBA: Mixture of Block Attention for Long-Context LLMs](http://arxiv.org/abs/2502.13189v1) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/MoonshotAI/MoBA)](https://github.com/MoonshotAI/MoBA) 
47. [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](http://arxiv.org/abs/2409.10516v3) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RetrievalAttention)](https://github.com/microsoft/RetrievalAttention) 
48. [SALS: Sparse Attention in Latent Space for KV cache Compression](http://arxiv.org/abs/2510.24273v1) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)]  
49. [SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training](http://arxiv.org/abs/2505.11594v1) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SageAttention)](https://github.com/thu-ml/SageAttention) 
50. [Spark Transformer: Reactivating Sparsity in FFN and Attention](http://arxiv.org/abs/2506.06644v2) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)]  
51. [Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation](http://arxiv.org/abs/2505.18875v3) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/svg-project/Sparse-VideoGen)](https://github.com/svg-project/Sparse-VideoGen) 
52. [Týr-the-Pruner: Unlocking Accurate 50% Structural Pruning for LLMs via Global Sparsity Distribution Optimization](http://arxiv.org/abs/2503.09657v2) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)]  
53. [VORTA: Efficient Video Diffusion via Routing Sparse Attention](http://arxiv.org/abs/2505.18809v2) [![Publish](https://img.shields.io/badge/2025-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/wenhao728/VORTA)](https://github.com/wenhao728/VORTA) 
54. [NanoFlow: Towards Optimal Large Language Model Serving Throughput](http://arxiv.org/abs/2408.12757v2) [![Publish](https://img.shields.io/badge/2025-OSDI-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/efeslab/Nanoflow)](https://github.com/efeslab/Nanoflow) 
55. [Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores](http://arxiv.org/abs/2501.09251v1) [![Publish](https://img.shields.io/badge/2025-PPoPP-green)]  
56. [PQCache: Product Quantization-based KVCache for Long Context LLM Inference](http://arxiv.org/abs/2407.12820v2) [![Publish](https://img.shields.io/badge/2025-SIGMOD-green)]  
57. [A Simple Linear Patch Revives Layer-Pruned Large Language Models](http://arxiv.org/abs/2505.24680v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
58. [Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching](http://arxiv.org/abs/2504.06319v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
59. [Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing](http://arxiv.org/abs/2505.19578v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
60. [Adaptive Computation Pruning for the Forgetting Transformer](http://arxiv.org/abs/2504.06949v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/zhixuan-lin/arctic-fox)](https://github.com/zhixuan-lin/arctic-fox) 
61. [Adaptive Layer-skipping in Pre-trained LLMs](http://arxiv.org/abs/2503.23798v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
62. [AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models](http://arxiv.org/abs/2506.03762v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
63. [Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models](http://arxiv.org/abs/2508.02128v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
64. [AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference](http://arxiv.org/abs/2502.04077v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
65. [Binary Quantization For LLMs Through Dynamic Grouping](http://arxiv.org/abs/2509.03054v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/johnnyzheng0636/WGM_bi_quan)](https://github.com/johnnyzheng0636/WGM_bi_quan) 
66. [CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs](http://arxiv.org/abs/2507.07145v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
67. [Characterizing Communication Patterns in Distributed Large Language Model Inference](http://arxiv.org/abs/2507.14392v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
68. [Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications](http://arxiv.org/abs/2507.03114v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
69. [ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](http://arxiv.org/abs/2502.00299v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
70. [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](http://arxiv.org/abs/2502.19811v3) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/flux)](https://github.com/bytedance/flux) 
71. [DBudgetKV: Dynamic Budget in KV Cache Compression for Ensuring Optimal Performance](http://arxiv.org/abs/2502.16886v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
72. [DReSS: Data-driven Regularized Structured Streamlining for Large Language Models](http://arxiv.org/abs/2501.17905v3) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
73. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](http://arxiv.org/abs/2501.12948v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-R1)](https://github.com/deepseek-ai/DeepSeek-R1) 
74. [DeltaLLM: A Training-Free Framework Exploiting Temporal Sparsity for Efficient Edge LLM Inference](http://arxiv.org/abs/2507.19608v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
75. [Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference](http://arxiv.org/abs/2511.15015v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
76. [Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need](http://arxiv.org/abs/2507.14397v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
77. [Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity](http://arxiv.org/abs/2502.11147v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
78. [EvolKV: Evolutionary KV Cache Compression for LLM Inference](http://arxiv.org/abs/2509.08315v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
79. [Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs](http://arxiv.org/abs/2502.06766v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ryansynk/topk-decoding)](https://github.com/ryansynk/topk-decoding) 
80. [Fast and Simplex: 2-Simplicial Attention in Triton](http://arxiv.org/abs/2507.02754v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
81. [FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation](http://arxiv.org/abs/2502.01068v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/dongwonjo/FastKV)](https://github.com/dongwonjo/FastKV) 
82. [Faster VGGT with Block-Sparse Global Attention](http://arxiv.org/abs/2509.07120v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
83. [Flash Sparse Attention: An Alternative Efficient Implementation of Native Sparse Attention Kernel](http://arxiv.org/abs/2508.18224v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/Relaxed-System-Lab/Flash-Sparse-Attention)](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention) 
84. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](http://arxiv.org/abs/2501.01005v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer)](https://github.com/flashinfer-ai/flashinfer) 
85. [FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension](http://arxiv.org/abs/2505.00570v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
86. [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](http://arxiv.org/abs/2508.06471v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/zai-org/GLM-4.5)](https://github.com/zai-org/GLM-4.5) 
87. [H1B-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference](http://arxiv.org/abs/2510.05529v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
88. [HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](http://arxiv.org/abs/2506.02572v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/gpzlx1/HATA)](https://github.com/gpzlx1/HATA) 
89. [HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs](http://arxiv.org/abs/2507.19823v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
90. [Hardware-Efficient Attention for Fast Decoding](http://arxiv.org/abs/2505.21487v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/grouped-latent-attention)](https://github.com/Dao-AILab/grouped-latent-attention) 
91. [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](http://arxiv.org/abs/2507.07120v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
92. [InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation](http://arxiv.org/abs/2509.24663v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/infllmv2_cuda_impl)](https://github.com/OpenBMB/infllmv2_cuda_impl) 
93. [Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation](http://arxiv.org/abs/2503.20552v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/Adrenaline)](https://github.com/ASISys/Adrenaline) 
94. [Instruction-Following Pruning for Large Language Models](http://arxiv.org/abs/2501.02086v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
95. [KV Cache Compression for Inference Efficiency in LLMs: A Review](http://arxiv.org/abs/2508.06297v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
96. [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](http://arxiv.org/abs/2502.16002v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/UCSB-NLP-Chang/KVLink)](https://github.com/UCSB-NLP-Chang/KVLink) 
97. [KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](http://arxiv.org/abs/2506.08018v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
98. [KeepKV: Eliminating Output Perturbation in KV Cache Compression for Efficient LLMs Inference](http://arxiv.org/abs/2504.09936v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
99. [LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation](http://arxiv.org/abs/2509.09754v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/MGDDestiny/Lava)](https://github.com/MGDDestiny/Lava) 
100. [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](http://arxiv.org/abs/2502.14866v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/omniserve)](https://github.com/mit-han-lab/omniserve) 
101. [LeanK: Learnable K Cache Channel Pruning for Efficient Decoding](http://arxiv.org/abs/2508.02215v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference)](https://github.com/microsoft/MInference) 
102. [LiteAttention: A Temporal Sparse Attention for Diffusion Transformers](http://arxiv.org/abs/2511.11062v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/moonmath-ai/LiteAttention)](https://github.com/moonmath-ai/LiteAttention) 
103. [MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving](http://arxiv.org/abs/2507.11507v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
104. [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](http://arxiv.org/abs/2505.11432v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
105. [MiniCPM4: Ultra-Efficient LLMs on End Devices](http://arxiv.org/abs/2506.07900v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/openbmb/minicpm)](https://github.com/openbmb/minicpm) 
106. [MiniMax-01: Scaling Foundation Models with Lightning Attention](http://arxiv.org/abs/2501.08313v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/MiniMax-AI/MiniMax-01)](https://github.com/MiniMax-AI/MiniMax-01) 
107. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](http://arxiv.org/abs/2506.13585v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/MiniMax-AI/MiniMax-M1)](https://github.com/MiniMax-AI/MiniMax-M1) 
108. [Mixture of Experts in Large Language Models](http://arxiv.org/abs/2507.11181v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
109. [Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing](http://arxiv.org/abs/2505.00315v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/piotrpiekos/MoSA)](https://github.com/piotrpiekos/MoSA) 
110. [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](http://arxiv.org/abs/2507.10524v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/raymin0223/mixture_of_recursions)](https://github.com/raymin0223/mixture_of_recursions) 
111. [MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](http://arxiv.org/abs/2511.15690v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
112. [MoPEQ: Mixture of Mixed Precision Quantized Experts](http://arxiv.org/abs/2509.02512v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/krishnateja95/MoE-Mixed-Prec)](https://github.com/krishnateja95/MoE-Mixed-Prec) 
113. [Mosaic: Composite Projection Pruning for Resource-efficient LLMs](http://arxiv.org/abs/2504.06323v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
114. [PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference](http://arxiv.org/abs/2509.04377v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
115. [Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs](http://arxiv.org/abs/2504.07866v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
116. [Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](http://arxiv.org/abs/2511.16156v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/OPPO-Mente-Lab/Qwen-Image-Pruning)](https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning) 
117. [PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention](http://arxiv.org/abs/2503.03588v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
118. [Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving](http://arxiv.org/abs/2503.00392v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/PSAttention)](https://github.com/ASISys/PSAttention) 
119. [Pruning General Large Language Models into Customized Expert Models](http://arxiv.org/abs/2506.02561v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/zhaoyiran924/Custom-Prune)](https://github.com/zhaoyiran924/Custom-Prune) 
120. [PureKV: Plug-and-Play KV Cache Optimization with Spatial-Temporal Sparse Attention for Vision-Language Large Models](http://arxiv.org/abs/2510.25600v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
121. [QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization](http://arxiv.org/abs/2506.22396v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
122. [Qwen3 Technical Report](http://arxiv.org/abs/2505.09388v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/QwenLM/Qwen3)](https://github.com/QwenLM/Qwen3) 
123. [R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration](http://arxiv.org/abs/2505.24133v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/Zefan-Cai/R-KV)](https://github.com/Zefan-Cai/R-KV) 
124. [Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation](http://arxiv.org/abs/2506.19852v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
125. [Rectified Sparse Attention](http://arxiv.org/abs/2506.04108v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/unilm)](https://github.com/microsoft/unilm) 
126. [Retrospective Sparse Attention for Efficient Long-Context Generation](http://arxiv.org/abs/2508.09001v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
127. [RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations](http://arxiv.org/abs/2501.16383v2) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
128. [SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling](http://arxiv.org/abs/2505.24179v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/BirdChristopher/SALE)](https://github.com/BirdChristopher/SALE) 
129. [SEAP: Training-free Sparse Expert Activation Pruning Unlock the Brainpower of Large Language Models](http://arxiv.org/abs/2503.07605v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/IAAR-Shanghai/SEAP)](https://github.com/IAAR-Shanghai/SEAP) 
130. [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](http://arxiv.org/abs/2509.24006v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SLA)](https://github.com/thu-ml/SLA) 
131. [SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space](http://arxiv.org/abs/2511.20102v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
132. [SeerAttention-R: Sparse Attention Adaptation for Long Reasoning](http://arxiv.org/abs/2506.08889v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SeerAttention)](https://github.com/microsoft/SeerAttention) 
133. [Seesaw: High-throughput LLM Inference via Model Re-sharding](http://arxiv.org/abs/2503.06433v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
134. [SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning](http://arxiv.org/abs/2508.06447v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
135. [Speed Always Wins: A Survey on Efficient Architectures for Large Language Models](http://arxiv.org/abs/2508.09834v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/weigao266/Awesome-Efficient-Arch)](https://github.com/weigao266/Awesome-Efficient-Arch) 
136. [SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](http://arxiv.org/abs/2507.06517v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/tyxqc/SpindleKV)](https://github.com/tyxqc/SpindleKV) 
137. [Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding](http://arxiv.org/abs/2507.19427v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
138. [Task-KV: Task-aware KV Cache Optimization via Semantic Differentiation of Attention Heads](http://arxiv.org/abs/2501.15113v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
139. [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](http://arxiv.org/abs/2504.17768v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/PiotrNawrot/sparse-frontier)](https://github.com/PiotrNawrot/sparse-frontier) 
140. [TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives](http://arxiv.org/abs/2503.20313v3) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Triton-distributed)](https://github.com/ByteDance-Seed/Triton-distributed) 
141. [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](http://arxiv.org/abs/2505.11329v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/tokenweave)](https://github.com/microsoft/tokenweave) 
142. [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](http://arxiv.org/abs/2504.19442v3) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Triton-distributed)](https://github.com/ByteDance-Seed/Triton-distributed) 
143. [Unveiling Super Experts in Mixture-of-Experts Large Language Models](http://arxiv.org/abs/2507.23279v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ZunhaiSu/Super-Experts-Profilling)](https://github.com/ZunhaiSu/Super-Experts-Profilling) 
144. [VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization](http://arxiv.org/abs/2510.06175v1) [![Publish](https://img.shields.io/badge/2025-arXiv-1E88E5)]  
145. [Attention-Gym: Triton-Based Sparse and Quantization Attention](https://github.com/RiseAI-Sys/attention-gym) [![Publish](https://img.shields.io/badge/2025-github-2F4F4F)] [![GitHub Repo stars](https://img.shields.io/github/stars/RiseAI-Sys/attention-gym)](https://github.com/RiseAI-Sys/attention-gym) 
146. [DeepEP: an efficient expert-parallel communication library](https://github.com/deepseek-ai/DeepEP) [![Publish](https://img.shields.io/badge/2025-github-2F4F4F)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepEP)](https://github.com/deepseek-ai/DeepEP) 
147. [Unified KV Cache Compression Methods for Auto-Regressive Models](https://github.com/Zefan-Cai/KVCache-Factory) [![Publish](https://img.shields.io/badge/2025-github-2F4F4F)] [![GitHub Repo stars](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory)](https://github.com/Zefan-Cai/KVCache-Factory) 
148. [kvpress: LLM KV cache compression made easy](https://github.com/NVIDIA/kvpress) [![Publish](https://img.shields.io/badge/2025-github-2F4F4F)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/kvpress)](https://github.com/NVIDIA/kvpress) 
</p>
</details>
<details open><summary><b>2024</b></summary> 
<p>

1. [Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983) [![Publish](https://img.shields.io/badge/2024-AAAI-FF4500)] [![GitHub Repo stars](https://img.shields.io/github/stars/CASIA-IVA-Lab/FLAP)](https://github.com/CASIA-IVA-Lab/FLAP) 
2. [$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens](http://arxiv.org/abs/2402.13718v3) [![Publish](https://img.shields.io/badge/2024-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/InfiniteBench)](https://github.com/OpenBMB/InfiniteBench) 
3. [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](http://arxiv.org/abs/2402.15220v4) [![Publish](https://img.shields.io/badge/2024-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/chunk-attention)](https://github.com/microsoft/chunk-attention) 
4. [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](http://arxiv.org/abs/2308.14508v2) [![Publish](https://img.shields.io/badge/2024-ACL-4169E1)]  
5. [Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning](https://dl.acm.org/doi/10.1145/3620666.3651379) [![Publish](https://img.shields.io/badge/2024-ASPLOS-9370DB)]  
6. [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](http://arxiv.org/abs/2401.16677v1) [![Publish](https://img.shields.io/badge/2024-ASPLOS-9370DB)]  
7. [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](http://arxiv.org/abs/2403.19708v3) [![Publish](https://img.shields.io/badge/2024-ATC-DC143C)]  
8. [A novel CUTLASS-based implementation of Tensor Parallelism for NVLink-enabled systems](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b) [![Publish](https://img.shields.io/badge/2024-Blog-696969)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/cutlass)](https://github.com/NVIDIA/cutlass) 
9. [[Distributed w/ TorchTitan] Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487/1) [![Publish](https://img.shields.io/badge/2024-Blog-696969)] [![GitHub Repo stars](https://img.shields.io/github/stars/pytorch/torchtitan)](https://github.com/pytorch/torchtitan) 
10. [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](http://arxiv.org/abs/2404.08763v4) [![Publish](https://img.shields.io/badge/2024-COLM-6495ED)] [![GitHub Repo stars](https://img.shields.io/github/stars/ScalingIntelligence/CATS)](https://github.com/ScalingIntelligence/CATS) 
11. [Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption](http://arxiv.org/abs/2407.18003v3) [![Publish](https://img.shields.io/badge/2024-COLM-6495ED)] [![GitHub Repo stars](https://img.shields.io/github/stars/zcli-charlie/Awesome-KV-Cache)](https://github.com/zcli-charlie/Awesome-KV-Cache) 
12. [SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference](http://arxiv.org/abs/2411.12692v1) [![Publish](https://img.shields.io/badge/2024-DATE-B22222)]  
13. [Post-Training Statistical Calibration for Higher Activation Sparsity](http://arxiv.org/abs/2412.07174v1) [![Publish](https://img.shields.io/badge/2024-ENLSP-00BFFF)] [![GitHub Repo stars](https://img.shields.io/github/stars/IntelLabs/SCAP)](https://github.com/IntelLabs/SCAP) 
14. [A Simple and Effective Pruning Approach for Large Language Models](http://arxiv.org/abs/2306.11695) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/locuslab/wanda)](https://github.com/locuslab/wanda) 
15. [Compressing LLMs: The Truth is Rarely Pure and Never Simple](http://arxiv.org/abs/2310.01382v2) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/llm-kick)](https://github.com/VITA-Group/llm-kick) 
16. [Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs](http://arxiv.org/abs/2310.08915v3) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/zyxxmu/DSnoT)](https://github.com/zyxxmu/DSnoT) 
17. [Efficient Streaming Language Models with Attention Sinks](http://arxiv.org/abs/2309.17453v4) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/streaming-llm)](https://github.com/mit-han-lab/streaming-llm) 
18. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/flash-attention)](https://github.com/Dao-AILab/flash-attention) 
19. [Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models](https://openreview.net/forum?id=Tr0lPx9woF) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/biomedical-cybernetics/Relative-importance-and-activation-pruning)](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning) 
20. [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/yuhuixu1993/qa-lora)](https://github.com/yuhuixu1993/qa-lora) 
21. [SAS: Structured Activation Spasification](https://openreview.net/forum?id=vZfi5to2Xl) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/DensoITLab/sas_)](https://github.com/DensoITLab/sas_) 
22. [SEA: Sparse Linear Attention with Estimated Attention Mask](http://arxiv.org/abs/2310.01777v2) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/gmlwns2000/sea-attention)](https://github.com/gmlwns2000/sea-attention) 
23. [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](http://arxiv.org/abs/2401.15024v2) [![Publish](https://img.shields.io/badge/2024-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/TransformerCompression)](https://github.com/microsoft/TransformerCompression) 
24. [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://arxiv.org/abs/2310.04564) [![Publish](https://img.shields.io/badge/2024-ICLR_oral-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/sjtu-ipads/powerinfer)](https://github.com/sjtu-ipads/powerinfer) 
25. [Accelerating Transformer Pre-training with 2:4 Sparsity](http://arxiv.org/abs/2404.01847v2) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/huyz2023/2by4-pretrain)](https://github.com/huyz2023/2by4-pretrain) 
26. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](http://arxiv.org/abs/2401.15077v2) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/SafeAILab/EAGLE)](https://github.com/SafeAILab/EAGLE) 
27. [FrameQuant: Flexible Low-Bit Quantization for Transformers](http://arxiv.org/abs/2403.06082v1) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/vsingh-group/FrameQuant)](https://github.com/vsingh-group/FrameQuant) 
28. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](http://arxiv.org/abs/2402.02750v2) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/jy-yuan/KIVI)](https://github.com/jy-yuan/KIVI) 
29. [LoRA+: Efficient Low Rank Adaptation of Large Models](http://arxiv.org/abs/2402.12354v1) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/nikhil-ghosh-berkeley/loraplus)](https://github.com/nikhil-ghosh-berkeley/loraplus) 
30. [OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization](http://arxiv.org/abs/2403.12983v1) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/mazumder-lab/OSSCAR)](https://github.com/mazumder-lab/OSSCAR) 
31. [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/pdf/2310.05175.pdf) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/luuyin/OWL)](https://github.com/luuyin/OWL) 
32. [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](http://arxiv.org/abs/2406.10774) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/quest)](https://github.com/mit-han-lab/quest) 
33. [SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models](http://arxiv.org/abs/2405.16057v1) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/Lucky-Lance/SPP)](https://github.com/Lucky-Lance/SPP) 
34. [SparQ Attention: Bandwidth-Efficient LLM Inference](http://arxiv.org/abs/2312.04985v5) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)]  
35. [Sparse is Enough in Fine-tuning Pre-trained Large Language Models](http://arxiv.org/abs/2312.11875v3) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/song-wx/SIFT)](https://github.com/song-wx/SIFT) 
36. [Sparse-IFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](http://arxiv.org/abs/2303.11525v3) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/CerebrasResearch/Sparse-IFT)](https://github.com/CerebrasResearch/Sparse-IFT) 
37. [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM)](https://github.com/SqueezeAILab/SqueezeLLM) 
38. [TinyTrain: Resource-Aware Task-Adaptive Sparse Training of DNNs at the Data-Scarce Edge](http://arxiv.org/abs/2307.09988v2) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/theyoungkwon/TinyTrain)](https://github.com/theyoungkwon/TinyTrain) 
39. [Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts](http://arxiv.org/abs/2403.08477v3) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/szc12153/sparse_interpolated_experts)](https://github.com/szc12153/sparse_interpolated_experts) 
40. [Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention](http://arxiv.org/abs/2405.17381v2) [![Publish](https://img.shields.io/badge/2024-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenNLPLab/TransnormerLLM)](https://github.com/OpenNLPLab/TransnormerLLM) 
41. [Splitwise: Efficient generative LLM inference using phase splitting](http://arxiv.org/abs/2311.18677v2) [![Publish](https://img.shields.io/badge/2024-ISCA-9932CC)] [![GitHub Repo stars](https://img.shields.io/github/stars/Mutinifni/splitwise-sim)](https://github.com/Mutinifni/splitwise-sim) 
42. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) [![Publish](https://img.shields.io/badge/2024-MLSys-DDA0DD)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/llm-awq)](https://github.com/mit-han-lab/llm-awq) 
43. [Vidur: A Large-Scale Simulation Framework For LLM Inference](http://arxiv.org/abs/2405.05465v2) [![Publish](https://img.shields.io/badge/2024-MLSys-DDA0DD)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/vidur)](https://github.com/microsoft/vidur) 
44. [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](http://arxiv.org/abs/2401.18079) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/SqueezeAILab/KVQuant)](https://github.com/SqueezeAILab/KVQuant) 
45. [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](http://arxiv.org/abs/2407.02490v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference)](https://github.com/microsoft/MInference) 
46. [MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models](http://arxiv.org/abs/2409.17481v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/MaskLLM)](https://github.com/NVlabs/MaskLLM) 
47. [SGLang: Efficient Execution of Structured Language Model Programs](http://arxiv.org/abs/2312.07104v2) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/sgl-project/sglang)](https://github.com/sgl-project/sglang) 
48. [SlimGPT: Layer-wise Structured Pruning for Large Language Models](http://arxiv.org/abs/2412.18110v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)]  
49. [SparseLLM: Towards Global Pruning for Pre-trained Language Models](http://arxiv.org/abs/2402.17946v3) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/BaiTheBest/SparseLLM)](https://github.com/BaiTheBest/SparseLLM) 
50. [ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification](http://arxiv.org/abs/2405.14256v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-FF1493)]  
51. [Fast and Effective Weight Update for Pruned Large Language Models](http://arxiv.org/abs/2401.02938v2) [![Publish](https://img.shields.io/badge/2024-TMLR-20B2AA)] [![GitHub Repo stars](https://img.shields.io/github/stars/fmfi-compbio/admm-pruning)](https://github.com/fmfi-compbio/admm-pruning) 
52. [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) [![Publish](https://img.shields.io/badge/2024-VLDB-A52A2A)] [![GitHub Repo stars](https://img.shields.io/github/stars/AlibabaResearch/flash-llm)](https://github.com/AlibabaResearch/flash-llm) 
53. [A Survey on Efficient Inference for Large Language Models](http://arxiv.org/abs/2404.14294v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
54. [A Survey on Inference Optimization Techniques for Mixture of Experts Models](http://arxiv.org/abs/2412.14219v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/MoE-Inf/awesome-moe-inference)](https://github.com/MoE-Inf/awesome-moe-inference) 
55. [A Survey on Large Language Model Acceleration based on KV Cache Management](http://arxiv.org/abs/2412.19442v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/TreeAI-Lab/Awesome-KV-Cache-Management)](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) 
56. [APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving](http://arxiv.org/abs/2411.17651v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/apex_plus)](https://github.com/microsoft/apex_plus) 
57. [AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis](http://arxiv.org/abs/2411.02117v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
58. [Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](http://arxiv.org/abs/2407.11550v3) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/FFY0/AdaKV)](https://github.com/FFY0/AdaKV) 
59. [Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs](http://arxiv.org/abs/2410.16135v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
60. [Beyond KV Caching: Shared Attention for Efficient LLMs](http://arxiv.org/abs/2407.12866v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/metacarbon/shareAtt)](https://github.com/metacarbon/shareAtt) 
61. [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2408.11796v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/Minitron)](https://github.com/NVlabs/Minitron) 
62. [CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation](http://arxiv.org/abs/2410.18311v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/wangqinsi1/CoreInfer)](https://github.com/wangqinsi1/CoreInfer) 
63. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](http://arxiv.org/abs/2405.04434v5) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V2)](https://github.com/deepseek-ai/DeepSeek-V2) 
64. [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V3)](https://github.com/deepseek-ai/DeepSeek-V3) 
65. [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](http://arxiv.org/abs/2401.06066v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-MoE)](https://github.com/deepseek-ai/DeepSeek-MoE) 
66. [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](http://arxiv.org/abs/2409.15241v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/deepspeedai/DeepSpeedExamples)](https://github.com/deepspeedai/DeepSpeedExamples) 
67. [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](http://arxiv.org/abs/2410.10819v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/duo-attention)](https://github.com/mit-han-lab/duo-attention) 
68. [Enabling High-Sparsity Foundational Llama Models with Efficient Pretraining and Deployment](http://arxiv.org/abs/2405.03594v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/neuralmagic/nm-vllm)](https://github.com/neuralmagic/nm-vllm) 
69. [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ldery/Bonsai)](https://github.com/ldery/Bonsai) 
70. [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](http://arxiv.org/abs/2406.06858v5) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/flux)](https://github.com/bytedance/flux) 
71. [FlashMask: Efficient and Rich Mask Extension of FlashAttention](http://arxiv.org/abs/2410.01359v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP)](https://github.com/PaddlePaddle/PaddleNLP) 
72. [GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM](http://arxiv.org/abs/2403.05527v4) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/HaoKang-Timmy/GEAR)](https://github.com/HaoKang-Timmy/GEAR) 
73. [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](http://arxiv.org/abs/2401.02669v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
74. [L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ](https://arxiv.org/abs/2402.04902) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
75. [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](http://arxiv.org/abs/2403.17919v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
76. [LLM Inference Serving: Survey of Recent Advances and Opportunities](http://arxiv.org/abs/2407.12391v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
77. [LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference](http://arxiv.org/abs/2407.14057v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
78. [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](http://arxiv.org/abs/2401.04658v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenNLPLab/lightning-attention)](https://github.com/OpenNLPLab/lightning-attention) 
79. [Massive Activations in Large Language Models](http://arxiv.org/abs/2402.17762v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/locuslab/massive-activations)](https://github.com/locuslab/massive-activations) 
80. [MiniCache: KV Cache Compression in Depth Dimension for Large Language Models](http://arxiv.org/abs/2405.14366v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/AkideLiu/MiniCache)](https://github.com/AkideLiu/MiniCache) 
81. [MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache](http://arxiv.org/abs/2411.18077) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/akshatsh49/MiniKV-Dev)](https://github.com/akshatsh49/MiniKV-Dev) 
82. [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](http://arxiv.org/abs/2404.02258v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
83. [MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression](http://arxiv.org/abs/2406.14909v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-nics/MoA)](https://github.com/thu-nics/MoA) 
84. [Multi-matrix Factorization Attention](http://arxiv.org/abs/2412.19255v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
85. [No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization](http://arxiv.org/abs/2402.18096v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
86. [Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification](http://arxiv.org/abs/2409.01366v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [Pytorch](https://anonymous.4open.science/r/CHESS-BA40/README.md) 
87. [Post-Training Sparse Attention with Double Sparsity](http://arxiv.org/abs/2408.07092v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/andy-yang-1/DoubleSparse)](https://github.com/andy-yang-1/DoubleSparse) 
88. [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](http://arxiv.org/abs/2406.06282v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [Website](https://powerinfer.ai/v2/) 
89. [PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization](http://arxiv.org/abs/2410.05265v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ChenMnZ/PrefixQuant)](https://github.com/ChenMnZ/PrefixQuant) 
90. [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/raincleared-song/sparse_gpu_operator)](https://github.com/raincleared-song/sparse_gpu_operator) 
91. [Q-Sparse: All Large Language Models can be Fully Sparsely-Activated](http://arxiv.org/abs/2407.10969v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
92. [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](http://arxiv.org/abs/2405.04532v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [Pytorch](https://hanlab.mit.edu/projects/qserve) 
93. [ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
94. [ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing](http://arxiv.org/abs/2412.14711v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/ReMoE)](https://github.com/thu-ml/ReMoE) 
95. [Recycled Attention: Efficient inference for long-context language models](http://arxiv.org/abs/2411.05787v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/carriex/recycled-attention)](https://github.com/carriex/recycled-attention) 
96. [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](http://arxiv.org/abs/2405.12981v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/JerryYin777/Cross-Layer-Attention)](https://github.com/JerryYin777/Cross-Layer-Attention) 
97. [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark](http://arxiv.org/abs/2402.11592v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ZO-Bench/ZO-LLM)](https://github.com/ZO-Bench/ZO-LLM) 
98. [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](http://arxiv.org/abs/2412.10319v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference)](https://github.com/microsoft/MInference) 
99. [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](http://arxiv.org/abs/2411.10958v6) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SageAttention)](https://github.com/thu-ml/SageAttention) 
100. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](http://arxiv.org/abs/2410.02367v8) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SageAttention)](https://github.com/thu-ml/SageAttention) 
101. [SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention](http://arxiv.org/abs/2406.15486v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
102. [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](http://arxiv.org/abs/2410.13276v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SeerAttention)](https://github.com/microsoft/SeerAttention) 
103. [ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models](http://arxiv.org/abs/2406.16635v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/abdelfattah-lab/shadow_llm)](https://github.com/abdelfattah-lab/shadow_llm) 
104. [SnapKV: LLM Knows What You are Looking for Before Generation](http://arxiv.org/abs/2404.14469v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/FasterDecoding/SnapKV)](https://github.com/FasterDecoding/SnapKV) 
105. [Transformers are Multi-State RNNs](http://arxiv.org/abs/2401.06104v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/schwartz-lab-NLP/TOVA)](https://github.com/schwartz-lab-NLP/TOVA) 
106. [Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters](http://arxiv.org/abs/2406.05955v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [Pytorch](https://huggingface.co/PowerInfer) 
107. [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](http://arxiv.org/abs/2411.15100v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/mlc-ai/xgrammar)](https://github.com/mlc-ai/xgrammar) 
108. [ZigZagkv: Dynamic KV Cache Compression for Long-context Modeling based on Layer Uncertainty](http://arxiv.org/abs/2412.09036v1) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
109. [ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification](http://arxiv.org/abs/2410.08584v2) [![Publish](https://img.shields.io/badge/2024-arXiv-1E88E5)]  
</p>
</details>
<details open><summary><b>2023</b></summary> 
<p>

1. [Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long Sequences](https://arxiv.org/abs/2210.11794) [![Publish](https://img.shields.io/badge/2023-AAAI-FF4500)] [![GitHub Repo stars](https://img.shields.io/github/stars/asFeng/Diffuser)](https://github.com/asFeng/Diffuser) 
2. [Gradient-based Intra-attention Pruning on Pre-trained Language Models](https://arxiv.org/abs/2212.07634) [![Publish](https://img.shields.io/badge/2023-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/airaria/GRAIN)](https://github.com/airaria/GRAIN) 
3. [Pruning Pre-trained Language Models Without Fine-Tuning](https://aclanthology.org/2023.acl-long.35.pdf) [![Publish](https://img.shields.io/badge/2023-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/kongds/SMP)](https://github.com/kongds/SMP) 
4. [Pruning Pre-trained Language Models with Principled Importance and Self-regularization](https://aclanthology.org/2023.findings-acl.573/) [![Publish](https://img.shields.io/badge/2023-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/drsy/pins)](https://github.com/drsy/pins) 
5. [Structured Pruning for Efficient Generative Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.692.pdf) [![Publish](https://img.shields.io/badge/2023-ACL-4169E1)]  
6. [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/abs/10.1145/3567955.3567959) [![Publish](https://img.shields.io/badge/2023-ASPLOS-9370DB)]  
7. [Structural Pruning of Large Language Models via Neural Architecture Search](https://openreview.net/forum?id=SHlZcInS6C) [![Publish](https://img.shields.io/badge/2023-AutoML_Workshop-A9A9A9)] [![GitHub Repo stars](https://img.shields.io/github/stars/awslabs/syne-tune)](https://github.com/awslabs/syne-tune) 
8. [Boost Vision Transformer with GPU-Friendly Sparsity and Quantization](http://arxiv.org/abs/2305.10727v1) [![Publish](https://img.shields.io/badge/2023-CVPR-2E8B57)]  
9. [SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer](https://arxiv.org/abs/2303.17605) [![Publish](https://img.shields.io/badge/2023-CVPR-2E8B57)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/sparsevit)](https://github.com/mit-han-lab/sparsevit) 
10. [TorchSparse++: Efficient Point Cloud Engine](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf) [![Publish](https://img.shields.io/badge/2023-CVPR_workshop-green)] [![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/torchsparse)](https://github.com/mit-han-lab/torchsparse) 
11. [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.10512.pdf) [![Publish](https://img.shields.io/badge/2023-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/QingruZhang/AdaLoRA)](https://github.com/QingruZhang/AdaLoRA) 
12. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323.pdf) [![Publish](https://img.shields.io/badge/2023-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/gptq)](https://github.com/IST-DASLab/gptq) 
13. [Minimum Variance Unbiased N:M Sparsity for the Neural Gradients](https://openreview.net/pdf?id=vuD2xEtxZcj) [![Publish](https://img.shields.io/badge/2023-ICLR-FF6B6B)]  
14. [The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers](https://openreview.net/forum?id=TJ2nxciYCk-) [![Publish](https://img.shields.io/badge/2023-ICLR-FF6B6B)]  
15. [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://openreview.net/forum?id=wIPIhHd00i) [![Publish](https://img.shields.io/badge/2023-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/FMInference/DejaVu)](https://github.com/FMInference/DejaVu) 
16. [SparseGPT: Massive Language Models Can be Accurately Pruned in one-shot.](https://arxiv.org/pdf/2301.00774.pdf) [![Publish](https://img.shields.io/badge/2023-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/sparsegpt)](https://github.com/IST-DASLab/sparsegpt) 
17. [Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) [![Publish](https://img.shields.io/badge/2023-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/yxli2123/LoSparse)](https://github.com/yxli2123/LoSparse) 
18. [Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning](https://proceedings.mlsys.org/paper_files/paper/2023/file/a10deb4d5227a8ea307ea8ff3cb712f4-Paper-mlsys2023.pdf) [![Publish](https://img.shields.io/badge/2023-MLSys-DDA0DD)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SparTA)](https://github.com/microsoft/SparTA) 
19. [ZipLM: Inference-Aware Structured Pruning of Language Models](https://openreview.net/pdf?id=bPFFPueAxm) [![Publish](https://img.shields.io/badge/2023-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/ZipLM)](https://github.com/IST-DASLab/ZipLM) 
20. [VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores](http://arxiv.org/abs/2310.02065v1) [![Publish](https://img.shields.io/badge/2023-SC-CD5C5C)] [![GitHub Repo stars](https://img.shields.io/github/stars/UDC-GAC/venom)](https://github.com/UDC-GAC/venom) 
21. [Efficient Memory Management for Large Language Model Serving with PagedAttention](http://arxiv.org/abs/2309.06180v1) [![Publish](https://img.shields.io/badge/2023-SOSP-8A2BE2)] [![GitHub Repo stars](https://img.shields.io/github/stars/vllm-project/vllm)](https://github.com/vllm-project/vllm) 
22. [Efficient Methods for Natural Language Processing: A Survey](https://arxiv.org/abs/2209.00099) [![Publish](https://img.shields.io/badge/2023-TACL-87CEEB)]  
23. [SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models](https://arxiv.org/abs/2303.10464) [![Publish](https://img.shields.io/badge/2023-UAI-green)]  
24. [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
25. [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
26. [Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models](http://arxiv.org/abs/2311.04902v2) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/VILA-Lab/GBLM-Pruner)](https://github.com/VILA-Lab/GBLM-Pruner) 
27. [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X](http://arxiv.org/abs/2303.17568v2) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/THUDM/CodeGeeX)](https://github.com/THUDM/CodeGeeX) 
28. [Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models](https://arxiv.org/abs/2310.05015) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/Moonlit)](https://github.com/microsoft/Moonlit) 
29. [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers](https://arxiv.org/abs/2305.15805) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/sanagno/adaptively_sparse_attention)](https://github.com/sanagno/adaptively_sparse_attention) 
30. [Efficient Guided Generation for Large Language Models](http://arxiv.org/abs/2307.09702v4) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
31. [Fine-Tuning Language Models with Just Forward Passes](http://arxiv.org/abs/2305.17333v3) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/MeZO)](https://github.com/princeton-nlp/MeZO) 
32. [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
33. [Gradient-Free Structured Pruning with Unlabeled Data](https://arxiv.org/abs/2303.04185) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
34. [H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](http://arxiv.org/abs/2306.14048) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/FMInference/H2O)](https://github.com/FMInference/H2O) 
35. [Knowledge-preserving Pruning for Pre-trained Language Models without Retraining](https://arxiv.org/abs/2308.03449) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
36. [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](http://arxiv.org/abs/2312.11514) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
37. [LLM-Pruner: On the Structural Pruning of Large Language Models](http://arxiv.org/abs/2305.11627v3) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/horseee/LLM-Pruner)](https://github.com/horseee/LLM-Pruner) 
38. [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
39. [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/yxli2123/LoftQ)](https://github.com/yxli2123/LoftQ) 
40. [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/OpenGVLab/OmniQuant)](https://github.com/OpenGVLab/OmniQuant) 
41. [Post-training Quantization for Neural Networks with Provable Guarantees](https://arxiv.org/pdf/2201.11113.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/YixuanSeanZhou/Quantized_Neural_Nets)](https://github.com/YixuanSeanZhou/Quantized_Neural_Nets) 
42. [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](http://arxiv.org/abs/2312.12456v1) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer)](https://github.com/SJTU-IPADS/PowerInfer) 
43. [Pruning Large Language Models via Accuracy Predictor](https://arxiv.org/abs/2309.09507) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
44. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/artidoro/qlora)](https://github.com/artidoro/qlora) 
45. [QuIP: Quantization with Incoherence Processing](https://arxiv.org/pdf/2307.13304.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/jerry-chee/QuIP)](https://github.com/jerry-chee/QuIP) 
46. [RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/pdf/2304.01089.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/hahnyuan/RPTQ4LLM)](https://github.com/hahnyuan/RPTQ4LLM) 
47. [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://xiamengzhou.github.io/sheared-llama/) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing)](https://github.com/princeton-nlp/LLM-Shearing) 
48. [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/pdf/2306.03078.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/Vahe1994/SpQR)](https://github.com/Vahe1994/SpQR) 
49. [Sparse Fine-tuning for Inference Acceleration of Large Language Models](https://arxiv.org/pdf/2310.06927.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/SparseFinetuning)](https://github.com/IST-DASLab/SparseFinetuning) 
50. [Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](https://arxiv.org/abs/2303.11525) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/CerebrasResearch/Sparse-IFT)](https://github.com/CerebrasResearch/Sparse-IFT) 
51. [Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging](https://arxiv.org/abs/2306.16788) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/ZIB-IOL/SMS)](https://github.com/ZIB-IOL/SMS) 
52. [Ten Lessons We Have Learned in the New Sparseland: A Short Handbook for Sparse Neural Network Researchers](https://arxiv.org/abs/2302.02596) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)]  
53. [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/essential_sparsity)](https://github.com/VITA-Group/essential_sparsity) 
54. [Training Transformers with 4-bit Integers](https://arxiv.org/abs//2306.11987) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/xijiu9/Train_Transformers_with_INT4)](https://github.com/xijiu9/Train_Transformers_with_INT4) 
55. [Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering](https://arxiv.org/abs/2304.12102) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/liyucheng09/Selective_Context)](https://github.com/liyucheng09/Selective_Context) 
56. [ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation](https://arxiv.org/abs/2303.08302) [![Publish](https://img.shields.io/badge/2023-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DeepSpeed)](https://github.com/microsoft/DeepSpeed) 
57. [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) [![Publish](https://img.shields.io/badge/2023-github-2F4F4F)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/FasterTransformer)](https://github.com/NVIDIA/FasterTransformer) 
</p>
</details>
<details open><summary><b>2022</b></summary> 
<p>

1. [Sparse Progressive Distillation: Resolving Overfitting under Pretrain-and-Finetune Paradigm](https://aclanthology.org/2022.acl-long.16/) [![Publish](https://img.shields.io/badge/2022-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/shaoyiHusky/SparseProgressiveDistillation)](https://github.com/shaoyiHusky/SparseProgressiveDistillation) 
2. [TextPruner: A Model Pruning Toolkit for Pre-Trained Language Models](https://arxiv.org/abs/2203.15996) [![Publish](https://img.shields.io/badge/2022-ACL-4169E1)] [![GitHub Repo stars](https://img.shields.io/github/stars/airaria/TextPruner)](https://github.com/airaria/TextPruner) 
3. [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](http://arxiv.org/abs/2105.05720v5) [![Publish](https://img.shields.io/badge/2022-ASPLOS-9370DB)] [![GitHub Repo stars](https://img.shields.io/github/stars/parasailteam/coconet)](https://github.com/parasailteam/coconet) 
4. [Creating Sparse GPT-3 Models with Iterative Pruning](https://www.cerebras.net/blog/creating-sparse-gpt-3-models-with-iterative-pruning) [![Publish](https://img.shields.io/badge/2022-Blog-696969)]  
5. [LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685) [![Publish](https://img.shields.io/badge/2022-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LoRA)](https://github.com/microsoft/LoRA) 
6. [SPDY: Accurate Pruning with Speedup Guarantees](https://arxiv.org/abs/2201.13096) [![Publish](https://img.shields.io/badge/2022-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/spdy)](https://github.com/IST-DASLab/spdy) 
7. [Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation](https://arxiv.org/abs/2209.00606) [![Publish](https://img.shields.io/badge/2022-MICRO-BA55D3)]  
8. [A Fast Post-Training Pruning Framework for Transformers](http://arxiv.org/abs/2204.09656v2) [![Publish](https://img.shields.io/badge/2022-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/WoosukKwon/retraining-free-pruning)](https://github.com/WoosukKwon/retraining-free-pruning) 
9. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) [![Publish](https://img.shields.io/badge/2022-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/flash-attention)](https://github.com/Dao-AILab/flash-attention) 
10. [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://openreview.net/pdf?id=ksVGCOlOEba) [![Publish](https://img.shields.io/badge/2022-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/OBC)](https://github.com/IST-DASLab/OBC) 
11. [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://openreview.net/forum?id=f-fVCElZ-G1) [![Publish](https://img.shields.io/badge/2022-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DeepSpeed)](https://github.com/microsoft/DeepSpeed) 
12. [Two Sparsities Are Better Than One: Unlocking the Performance Benefits of Sparse-Sparse Networks](https://iopscience.iop.org/article/10.1088/2634-4386/ac7c8a) [![Publish](https://img.shields.io/badge/2022-Neuromorphic_Computing_and_Engineering-40E0D0)]  
13. [Transformer Acceleration with Dynamic Sparse Attention](http://arxiv.org/abs/2110.11299v1) [![Publish](https://img.shields.io/badge/2022-TC-F08080)]  
14. [An Algorithm-Hardware Co-Optimized Framework for Accelerating N:M Sparse Transformers](https://arxiv.org/abs/2208.06118) [![Publish](https://img.shields.io/badge/2022-VLSI-8B0000)]  
15. [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) [![Publish](https://img.shields.io/badge/2022-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/neuralmagic/sparseml)](https://github.com/neuralmagic/sparseml) 
</p>
</details>
<details open><summary><b>2021</b></summary> 
<p>

1. [Post-training deep neural network pruning via layer-wise calibration](https://openaccess.thecvf.com/content/ICCV2021W/LPCV/papers/Lazarevich_Post-Training_Deep_Neural_Network_Pruning_via_Layer-Wise_Calibration_ICCVW_2021_paper.pdf) [![Publish](https://img.shields.io/badge/2021-ICCV-green)]  
2. [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://openreview.net/pdf?id=POWv6hDd9XH) [![Publish](https://img.shields.io/badge/2021-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/yhhhli/BRECQ)](https://github.com/yhhhli/BRECQ) 
3. [Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch](https://openreview.net/forum?id=K9bw7vqp_s) [![Publish](https://img.shields.io/badge/2021-ICLR-FF6B6B)] [![GitHub Repo stars](https://img.shields.io/github/stars/aojunzz/NM-sparsity)](https://github.com/aojunzz/NM-sparsity) 
4. [A Greedy Algorithm for Quantizing Neural Networks](https://jmlr.csail.mit.edu/papers/volume22/20-1233/20-1233.pdf) [![Publish](https://img.shields.io/badge/2021-JMLR-008B8B)] [![GitHub Repo stars](https://img.shields.io/github/stars/YixuanSeanZhou/Quantized_Neural_Nets)](https://github.com/YixuanSeanZhou/Quantized_Neural_Nets) 
5. [Channel Permutations for N:M Sparsity](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) [![Publish](https://img.shields.io/badge/2021-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/apex)](https://github.com/NVIDIA/apex) 
6. [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) [![Publish](https://img.shields.io/badge/2021-arXiv-1E88E5)]  
7. [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554) [![Publish](https://img.shields.io/badge/2021-arXiv-1E88E5)]  
</p>
</details>
<details open><summary><b>2020</b></summary> 
<p>

1. [Fast Sparse ConvNets](https://openaccess.thecvf.com/content_CVPR_2020/papers/Elsen_Fast_Sparse_ConvNets_CVPR_2020_paper.pdf) [![Publish](https://img.shields.io/badge/2020-CVPR-2E8B57)] [![GitHub Repo stars](https://img.shields.io/github/stars/fastconvnets/cvpr2020)](https://github.com/fastconvnets/cvpr2020) 
2. [Inducing and Exploiting Activation Sparsity for Fast Neural Network Inference](http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf) [![Publish](https://img.shields.io/badge/2020-ICML-FF8C00)]  
3. [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) [![Publish](https://img.shields.io/badge/2020-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/block_movement_pruning)](https://github.com/huggingface/block_movement_pruning) 
4. [GPU Kernels for Block-Sparse Weights](https://cdn.openai.com/blocksparse/blocksparsepaper.pdf) [![Publish](https://img.shields.io/badge/2020-arXiv-1E88E5)] [![GitHub Repo stars](https://img.shields.io/github/stars/openai/blocksparse)](https://github.com/openai/blocksparse) 
</p>
</details>
<details open><summary><b>2019</b></summary> 
<p>

1. [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129) [![Publish](https://img.shields.io/badge/2019-ICML-FF8C00)] [![GitHub Repo stars](https://img.shields.io/github/stars/ucbrise/actnn)](https://github.com/ucbrise/actnn) 
</p>
</details>
<details open><summary><b>2018</b></summary> 
<p>

1. [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294) [![Publish](https://img.shields.io/badge/2018-ECCV-3CB371)] [![GitHub Repo stars](https://img.shields.io/github/stars/bzantium/pytorch-admm-pruning)](https://github.com/bzantium/pytorch-admm-pruning) 
</p>
</details>
<details open><summary><b>2017</b></summary> 
<p>

1. [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/pdf/1607.04381.pdf) [![Publish](https://img.shields.io/badge/2017-ICLR-FF6B6B)]  
2. [Attention Is All You Need](http://arxiv.org/abs/1706.03762v7) [![Publish](https://img.shields.io/badge/2017-NeurIPS-FF1493)]  
3. [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/pdf/1705.07565.pdf) [![Publish](https://img.shields.io/badge/2017-NeurIPS-FF1493)] [![GitHub Repo stars](https://img.shields.io/github/stars/csyhhu/L-OBS)](https://github.com/csyhhu/L-OBS) 
</p>
</details>
<details open><summary><b>2016</b></summary> 
<p>

1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf) [![Publish](https://img.shields.io/badge/2016-ICLR-FF6B6B)]  
</p>
</details>
<details open><summary><b>1993</b></summary> 
<p>

1. [Optimal Brain Surgeon and general network pruning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=298572&tag=1) [![Publish](https://img.shields.io/badge/1993--green)]  
</p>
</details>
<details open><summary><b>1989</b></summary> 
<p>

1. [Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [![Publish](https://img.shields.io/badge/1989-NeurIPS-FF1493)]  
</p>
</details>
</p>
</details>

## References

1. [https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) [[![GitHub Repo stars](https://img.shields.io/github/stars/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)]

2. [https://github.com/weigao266/Awesome-Efficient-Arch](https://github.com/weigao266/Awesome-Efficient-Arch) [[![GitHub Repo stars](https://img.shields.io/github/stars/weigao266/Awesome-Efficient-Arch)](https://github.com/weigao266/Awesome-Efficient-Arch)]

3. [https://github.com/horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) [[![GitHub Repo stars](https://img.shields.io/github/stars/horseee/Awesome-Efficient-LLM)](https://github.com/horseee/Awesome-Efficient-LLM)]

4. [https://github.com/DefTruth/Awesome-Diffusion-Inference](https://github.com/DefTruth/Awesome-Diffusion-Inference) [[![GitHub Repo stars](https://img.shields.io/github/stars/DefTruth/Awesome-Diffusion-Inference)](https://github.com/DefTruth/Awesome-Diffusion-Inference)]

5. [https://github.com/DefTruth/Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) [[![GitHub Repo stars](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference)](https://github.com/DefTruth/Awesome-LLM-Inference)]

6. [https://github.com/AmberLJC/LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) [[![GitHub Repo stars](https://img.shields.io/github/stars/AmberLJC/LLMSys-PaperList)](https://github.com/AmberLJC/LLMSys-PaperList)]

7. [https://github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) [[![GitHub Repo stars](https://img.shields.io/github/stars/Hannibal046/Awesome-LLM)](https://github.com/Hannibal046/Awesome-LLM)]

8. [https://github.com/AmadeusChan/Awesome-LLM-System-Papers](https://github.com/AmadeusChan/Awesome-LLM-System-Papers) [[![GitHub Repo stars](https://img.shields.io/github/stars/AmadeusChan/Awesome-LLM-System-Papers)](https://github.com/AmadeusChan/Awesome-LLM-System-Papers)]

9. [https://github.com/KnowingNothing/compiler-and-arch](https://github.com/KnowingNothing/compiler-and-arch) [[![GitHub Repo stars](https://img.shields.io/github/stars/KnowingNothing/compiler-and-arch)](https://github.com/KnowingNothing/compiler-and-arch)]

10. [https://papercopilot.com/paper-list](https://papercopilot.com/paper-list)

11. [https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) [[![GitHub Repo stars](https://img.shields.io/github/stars/TreeAI-Lab/Awesome-KV-Cache-Management)](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)]

12. [https://github.com/October2001/Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) [[![GitHub Repo stars](https://img.shields.io/github/stars/October2001/Awesome-KV-Cache-Compression)](https://github.com/October2001/Awesome-KV-Cache-Compression)]

13. [https://github.com/he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning) [[![GitHub Repo stars](https://img.shields.io/github/stars/he-y/Awesome-Pruning)](https://github.com/he-y/Awesome-Pruning)]

14. [https://github.com/htqin/awesome-model-quantization](https://github.com/htqin/awesome-model-quantization) [[![GitHub Repo stars](https://img.shields.io/github/stars/htqin/awesome-model-quantization)](https://github.com/htqin/awesome-model-quantization)]

15. [https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression) [[![GitHub Repo stars](https://img.shields.io/github/stars/csyhhu/Awesome-Deep-Neural-Network-Compression)](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression)]

16. [https://github.com/AojunZhou/Efficient-Deep-Learning](https://github.com/AojunZhou/Efficient-Deep-Learning) [[![GitHub Repo stars](https://img.shields.io/github/stars/AojunZhou/Efficient-Deep-Learning)](https://github.com/AojunZhou/Efficient-Deep-Learning)]

17. [https://github.com/chester256/Model-Compression-Papers](https://github.com/chester256/Model-Compression-Papers) [[![GitHub Repo stars](https://img.shields.io/github/stars/chester256/Model-Compression-Papers)](https://github.com/chester256/Model-Compression-Papers)]

