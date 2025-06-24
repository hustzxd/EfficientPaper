# EfficientPaper
Pruning, Quantization and efficient-inference/training paper list.

## Table of Contents
- [EfficientPaper](#efficientpaper)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [:sparkles: Paper List](#paper-list)
    - [keyword](cls_keyword.md)
    - [year](cls_year.md)
    - [publication](cls_publication.md)
    - [institution](cls_institution.md)
    - [author](cls_author.md)
  - [References](#references)



## Getting Started
```bash
pip install protobuf==5.27.2 pandas arxiv 
```
1. Add paper information by `./add_paper_info.sh`
2. Run `./refresh_readme.sh`

<details><summary><b>sparsegpt.prototxt</b></summary>	
<p>

```
paper {
  title: "SparseGPT: Massive Language Models Can be Accurately Pruned in one-shot."
  abbr: "SparseGPT"
  url: "https://arxiv.org/pdf/2301.00774.pdf"
  authors: "Elias Frantar"
  authors: "Dan Alistarh"
  institutions: "IST Austria"
  institutions: "Neural Magic"
}
pub {
  where: "arXiv"
  year: 2023
}
code {
  type: "Pytorch"
  url: "https://github.com/IST-DASLab/sparsegpt"
}
note {
  url: "SparseGPT.md"
}
keyword {
  words: sparse_pruning
}
```

</p>
</details>




## Paper List

<details open><summary>

### year
</summary> 
<p>

<details open><summary><b>2025</b></summary> 
<p>

1. [AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference](http://arxiv.org/abs/2501.02336v1) [![Publish](https://img.shields.io/badge/2025-AAAI-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/AdaSkip) 
2. [Pruning Large Language Models with Semi-Structural Adaptive Sparse Training](http://arxiv.org/abs/2407.20584v3) [![Publish](https://img.shields.io/badge/2025-AAAI-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/Adaptive-Sparse-Trainer) 
3. [COMET: Towards Partical W4A4KV4 LLMs Serving](http://arxiv.org/abs/2410.12168v1) [![Publish](https://img.shields.io/badge/2025-ASPLOS-orange)]  
4. [POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference](http://arxiv.org/abs/2410.18038v2) [![Publish](https://img.shields.io/badge/2025-ASPLOS-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/vattention) 
5. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](http://arxiv.org/abs/2405.04437v3) [![Publish](https://img.shields.io/badge/2025-ASPLOS-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/vattention) 
6. [Enhancing One-shot Pruned Pre-trained Language Models through Sparse-Dense-Sparse Mechanism](http://arxiv.org/abs/2408.10473v1) [![Publish](https://img.shields.io/badge/2025-Coling-green)]  
7. [FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference](http://arxiv.org/abs/2502.20766v1) [![Publish](https://img.shields.io/badge/2025-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/FlexPrefill) 
8. [Forgetting Transformer: Softmax Attention with a Forget Gate](http://arxiv.org/abs/2503.02130v2) [![Publish](https://img.shields.io/badge/2025-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/zhixuan-lin/forgetting-transformer) 
9. [R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](http://arxiv.org/abs/2504.19449v1) [![Publish](https://img.shields.io/badge/2025-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/R-Sparse) 
10. [ReAttention: Training-Free Infinite Context with Finite Attention Scope](http://arxiv.org/abs/2407.15176v3) [![Publish](https://img.shields.io/badge/2025-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/OpenMOSS/ReAttention) 
11. [Training-Free Activation Sparsity in Large Language Models](http://arxiv.org/abs/2408.14690v1) [![Publish](https://img.shields.io/badge/2025-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/FasterDecoding/TEAL) 
12. [Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity](http://arxiv.org/abs/2412.02252v1) [![Publish](https://img.shields.io/badge/2025-ICML-blue)]  
13. [XAttention: Block Sparse Attention with Antidiagonal Scoring](http://arxiv.org/abs/2503.16428v1) [![Publish](https://img.shields.io/badge/2025-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/x-attention) 
14. [SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting](http://arxiv.org/abs/2504.08850v1) [![Publish](https://img.shields.io/badge/2025-ISCA-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/infinigence/SpecEE) 
15. [A Simple Linear Patch Revives Layer-Pruned Large Language Models](http://arxiv.org/abs/2505.24680v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
16. [Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores](http://arxiv.org/abs/2501.09251v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
17. [Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching](http://arxiv.org/abs/2504.06319v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
18. [Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing](http://arxiv.org/abs/2505.19578v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
19. [AdaSplash: Adaptive Sparse Flash Attention](http://arxiv.org/abs/2502.12082v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/deep-spin/adasplash) 
20. [Adaptive Computation Pruning for the Forgetting Transformer](http://arxiv.org/abs/2504.06949v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/zhixuan-lin/arctic-fox) 
21. [Adaptive Layer-skipping in Pre-trained LLMs](http://arxiv.org/abs/2503.23798v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
22. [AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference](http://arxiv.org/abs/2502.04077v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
23. [BaWA: Automatic Optimizing Pruning Metric for Large Language Models with Balanced Weight and Activation](https://openreview.net/forum?id=YrCvW1Hx7g) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
24. [ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](http://arxiv.org/abs/2502.00299v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
25. [DBudgetKV: Dynamic Budget in KV Cache Compression for Ensuring Optimal Performance](http://arxiv.org/abs/2502.16886v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
26. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](http://arxiv.org/abs/2501.12948v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-R1) 
27. [Delta Attention: Fast and Accurate Sparse Attention Inference by Delta Correction](http://arxiv.org/abs/2505.11254v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
28. [Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity](http://arxiv.org/abs/2502.11147v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
29. [Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs](http://arxiv.org/abs/2502.06766v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ryansynk/topk-decoding) 
30. [FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation](http://arxiv.org/abs/2502.01068v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/dongwonjo/FastKV) 
31. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](http://arxiv.org/abs/2501.01005v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer) 
32. [FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension](http://arxiv.org/abs/2505.00570v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
33. [HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](http://arxiv.org/abs/2506.02572v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/gpzlx1/HATA) 
34. [Hardware-Efficient Attention for Fast Decoding](http://arxiv.org/abs/2505.21487v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/grouped-latent-attention) 
35. [Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation](http://arxiv.org/abs/2503.20552v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/Adrenaline) 
36. [Instruction-Following Pruning for Large Language Models](http://arxiv.org/abs/2501.02086v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
37. [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](http://arxiv.org/abs/2502.16002v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/UCSB-NLP-Chang/KVLink) 
38. [KeepKV: Eliminating Output Perturbation in KV Cache Compression for Efficient LLMs Inference](http://arxiv.org/abs/2504.09936v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
39. [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](http://arxiv.org/abs/2502.14866v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/omniserve) 
40. [MiniCPM4: Ultra-Efficient LLMs on End Devices](http://arxiv.org/abs/2506.07900v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/openbmb/minicpm) 
41. [Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing](http://arxiv.org/abs/2505.00315v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/piotrpiekos/MoSA) 
42. [MoBA: Mixture of Block Attention for Long-Context LLMs](http://arxiv.org/abs/2502.13189v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/MoonshotAI/MoBA) 
43. [Mosaic: Composite Projection Pruning for Resource-efficient LLMs](http://arxiv.org/abs/2504.06323v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
44. [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](http://arxiv.org/abs/2502.11089v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
45. [PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention](http://arxiv.org/abs/2503.03588v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
46. [Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving](http://arxiv.org/abs/2503.00392v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ASISys/PSAttention) 
47. [Pruning General Large Language Models into Customized Expert Models](http://arxiv.org/abs/2506.02561v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/zhaoyiran924/Custom-Prune) 
48. [R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration](http://arxiv.org/abs/2505.24133v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/Zefan-Cai/R-KV) 
49. [Rectified Sparse Attention](http://arxiv.org/abs/2506.04108v2) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/unilm) 
50. [Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving](http://arxiv.org/abs/2503.24000v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/LLMkvsys/rethink-kv-compression) 
51. [SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling](http://arxiv.org/abs/2505.24179v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/BirdChristopher/SALE) 
52. [SEAP: Training-free Sparse Expert Activation Pruning Unlock the Brainpower of Large Language Models](http://arxiv.org/abs/2503.07605v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/IAAR-Shanghai/SEAP) 
53. [SeerAttention-R: Sparse Attention Adaptation for Long Reasoning](http://arxiv.org/abs/2506.08889v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SeerAttention) 
54. [SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](http://arxiv.org/abs/2502.18137v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/SpargeAttn) 
55. [Task-KV: Task-aware KV Cache Optimization via Semantic Differentiation of Attention Heads](http://arxiv.org/abs/2501.15113v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)]  
56. [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](http://arxiv.org/abs/2504.17768v1) [![Publish](https://img.shields.io/badge/2025-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/PiotrNawrot/sparse-frontier) 
57. [Unified KV Cache Compression Methods for Auto-Regressive Models](https://github.com/Zefan-Cai/KVCache-Factory) [![Publish](https://img.shields.io/badge/2025-github-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory) 
58. kvpress [![Publish](https://img.shields.io/badge/2025-github-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/kvpress) 
</p>
</details>
<details open><summary><b>2024</b></summary> 
<p>

1. [Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983) [![Publish](https://img.shields.io/badge/2024-AAAI-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/CASIA-IVA-Lab/FLAP) 
2. [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](http://arxiv.org/abs/2402.15220v4) [![Publish](https://img.shields.io/badge/2024-ACL-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/chunk-attention) 
3. [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](http://arxiv.org/abs/2403.19708v3) [![Publish](https://img.shields.io/badge/2024-ATC-orange)]  
4. [CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models](http://arxiv.org/abs/2404.08763v4) [![Publish](https://img.shields.io/badge/2024-COLM-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/ScalingIntelligence/CATS) 
5. [Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption](http://arxiv.org/abs/2407.18003v3) [![Publish](https://img.shields.io/badge/2024-COLM-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/zcli-charlie/Awesome-KV-Cache) 
6. [SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference](http://arxiv.org/abs/2411.12692v1) [![Publish](https://img.shields.io/badge/2024-DATE-orange)]  
7. [Post-Training Statistical Calibration for Higher Activation Sparsity](http://arxiv.org/abs/2412.07174v1) [![Publish](https://img.shields.io/badge/2024-ENLSP-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/IntelLabs/SCAP) 
8. [A Simple and Effective Pruning Approach for Large Language Models](http://arxiv.org/abs/2306.11695) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/locuslab/wanda) 
9. [Compressing LLMs: The Truth is Rarely Pure and Never Simple](http://arxiv.org/abs/2310.01382v2) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/llm-kick) 
10. [Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs](http://arxiv.org/abs/2310.08915v3) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/zyxxmu/DSnoT) 
11. [Efficient Streaming Language Models with Attention Sinks](http://arxiv.org/abs/2309.17453v4) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/streaming-llm) 
12. [Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models](https://openreview.net/forum?id=Tr0lPx9woF) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/biomedical-cybernetics/Relative-importance-and-activation-pruning) 
13. [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/yuhuixu1993/qa-lora) 
14. [SAS: Structured Activation Spasification](https://openreview.net/forum?id=vZfi5to2Xl) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/DensoITLab/sas_) 
15. [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](http://arxiv.org/abs/2401.15024v2) [![Publish](https://img.shields.io/badge/2024-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/TransformerCompression) 
16. [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://arxiv.org/abs/2310.04564) [![Publish](https://img.shields.io/badge/2024-ICLR_oral-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/sjtu-ipads/powerinfer) 
17. [Accelerating Transformer Pre-training with 2:4 Sparsity](http://arxiv.org/abs/2404.01847v2) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/huyz2023/2by4-pretrain) 
18. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](http://arxiv.org/abs/2401.15077v2) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/SafeAILab/EAGLE) 
19. [FrameQuant: Flexible Low-Bit Quantization for Transformers](http://arxiv.org/abs/2403.06082v1) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/vsingh-group/FrameQuant) 
20. [LoRA+: Efficient Low Rank Adaptation of Large Models](http://arxiv.org/abs/2402.12354v1) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/nikhil-ghosh-berkeley/loraplus) 
21. [OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization](http://arxiv.org/abs/2403.12983v1) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/mazumder-lab/OSSCAR) 
22. [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/pdf/2310.05175.pdf) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/luuyin/OWL) 
23. [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](http://arxiv.org/abs/2406.10774) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/quest) 
24. [SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models](http://arxiv.org/abs/2405.16057v1) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/Lucky-Lance/SPP) 
25. [SparQ Attention: Bandwidth-Efficient LLM Inference](http://arxiv.org/abs/2312.04985v5) [![Publish](https://img.shields.io/badge/2024-ICML-blue)]  
26. [Sparse is Enough in Fine-tuning Pre-trained Large Language Models](http://arxiv.org/abs/2312.11875v3) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/song-wx/SIFT) 
27. [Sparse-IFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](http://arxiv.org/abs/2303.11525v3) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/CerebrasResearch/Sparse-IFT) 
28. [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629v4) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM) 
29. [TinyTrain: Resource-Aware Task-Adaptive Sparse Training of DNNs at the Data-Scarce Edge](http://arxiv.org/abs/2307.09988v2) [![Publish](https://img.shields.io/badge/2024-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/theyoungkwon/TinyTrain) 
30. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) [![Publish](https://img.shields.io/badge/2024-MLSys-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/llm-awq) 
31. [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](http://arxiv.org/abs/2407.02490v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference) 
32. [MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models](http://arxiv.org/abs/2409.17481v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/MaskLLM) 
33. [SGLang: Efficient Execution of Structured Language Model Programs](http://arxiv.org/abs/2312.07104v2) [![Publish](https://img.shields.io/badge/2024-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/sgl-project/sglang) 
34. [SlimGPT: Layer-wise Structured Pruning for Large Language Models](http://arxiv.org/abs/2412.18110v1) [![Publish](https://img.shields.io/badge/2024-NeurIPS-blue)]  
35. [SparseLLM: Towards Global Pruning for Pre-trained Language Models](http://arxiv.org/abs/2402.17946v3) [![Publish](https://img.shields.io/badge/2024-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/BaiTheBest/SparseLLM) 
36. [Fast and Effective Weight Update for Pruned Large Language Models](http://arxiv.org/abs/2401.02938v2) [![Publish](https://img.shields.io/badge/2024-TMLR-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/fmfi-compbio/admm-pruning) 
37. [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) [![Publish](https://img.shields.io/badge/2024-VLDB-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/AlibabaResearch/flash-llm) 
38. [A Survey on Efficient Inference for Large Language Models](http://arxiv.org/abs/2404.14294v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
39. [A Survey on Large Language Model Acceleration based on KV Cache Management](http://arxiv.org/abs/2412.19442v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/TreeAI-Lab/Awesome-KV-Cache-Management) 
40. [AVSS: Layer Importance Evaluation in Large Language Models via Activation Variance-Sparsity Analysis](http://arxiv.org/abs/2411.02117v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
41. [Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](http://arxiv.org/abs/2407.11550v3) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/FFY0/AdaKV) 
42. [Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs](http://arxiv.org/abs/2410.16135v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
43. [Beyond KV Caching: Shared Attention for Efficient LLMs](http://arxiv.org/abs/2407.12866v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/metacarbon/shareAtt) 
44. [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2408.11796v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/Minitron) 
45. [CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation](http://arxiv.org/abs/2410.18311v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/wangqinsi1/CoreInfer) 
46. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](http://arxiv.org/abs/2405.04434v5) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V2) 
47. [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V3) 
48. [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](http://arxiv.org/abs/2401.06066v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-MoE) 
49. [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](http://arxiv.org/abs/2410.10819v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/duo-attention) 
50. [Enabling High-Sparsity Foundational Llama Models with Efficient Pretraining and Deployment](http://arxiv.org/abs/2405.03594v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/neuralmagic/nm-vllm) 
51. [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ldery/Bonsai) 
52. [FlashMask: Efficient and Rich Mask Extension of FlashAttention](http://arxiv.org/abs/2410.01359v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP) 
53. [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](http://arxiv.org/abs/2401.02669v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
54. [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](http://arxiv.org/abs/2401.18079v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/SqueezeAILab/KVQuant) 
55. [L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ](https://arxiv.org/abs/2402.04902) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
56. [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](http://arxiv.org/abs/2403.17919v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
57. [Massive Activations in Large Language Models](http://arxiv.org/abs/2402.17762v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/locuslab/massive-activations) 
58. [MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache](http://arxiv.org/abs/2411.18077v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
59. [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](http://arxiv.org/abs/2404.02258v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
60. [MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression](http://arxiv.org/abs/2406.14909v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/thu-nics/MoA) 
61. [Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification](http://arxiv.org/abs/2409.01366v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] [Pytorch](https://anonymous.4open.science/r/CHESS-BA40/README.md) 
62. [Post-Training Sparse Attention with Double Sparsity](http://arxiv.org/abs/2408.07092v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/andy-yang-1/DoubleSparse) 
63. [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](http://arxiv.org/abs/2406.06282v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] [Website](https://powerinfer.ai/v2/) 
64. [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/raincleared-song/sparse_gpu_operator) 
65. [Q-Sparse: All Large Language Models can be Fully Sparsely-Activated](http://arxiv.org/abs/2407.10969v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
66. [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](http://arxiv.org/abs/2405.04532v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] [Pytorch](https://hanlab.mit.edu/projects/qserve) 
67. [ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
68. [ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing](http://arxiv.org/abs/2412.14711v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/thu-ml/ReMoE) 
69. [Recycled Attention: Efficient inference for long-context language models](http://arxiv.org/abs/2411.05787v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/carriex/recycled-attention) 
70. [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](http://arxiv.org/abs/2405.12981v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/JerryYin777/Cross-Layer-Attention) 
71. [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark](http://arxiv.org/abs/2402.11592v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZO-Bench/ZO-LLM) 
72. [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](http://arxiv.org/abs/2412.10319v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference) 
73. [SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention](http://arxiv.org/abs/2406.15486v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
74. [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](http://arxiv.org/abs/2410.13276v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SeerAttention) 
75. [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](http://arxiv.org/abs/2410.21465v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/ShadowKV) 
76. [ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models](http://arxiv.org/abs/2406.16635v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/abdelfattah-lab/shadow_llm) 
77. [SnapKV: LLM Knows What You are Looking for Before Generation](http://arxiv.org/abs/2404.14469v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/FasterDecoding/SnapKV) 
78. [Sparsing Law: Towards Large Language Models with Greater Activation Sparsity](http://arxiv.org/abs/2411.02335v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/thunlp/SparsingLaw) 
79. [Star Attention: Efficient LLM Inference over Long Sequences](http://arxiv.org/abs/2411.17116v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/Star-Attention) 
80. [Transformers are Multi-State RNNs](http://arxiv.org/abs/2401.06104v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/schwartz-lab-NLP/TOVA) 
81. [Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters](http://arxiv.org/abs/2406.05955v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] [Pytorch](https://huggingface.co/PowerInfer) 
82. [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](http://arxiv.org/abs/2411.15100v2) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/mlc-ai/xgrammar) 
83. [ZigZagkv: Dynamic KV Cache Compression for Long-context Modeling based on Layer Uncertainty](http://arxiv.org/abs/2412.09036v1) [![Publish](https://img.shields.io/badge/2024-arXiv-violet)]  
</p>
</details>
<details open><summary><b>2023</b></summary> 
<p>

1. [Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long Sequences](https://arxiv.org/abs/2210.11794) [![Publish](https://img.shields.io/badge/2023-AAAI-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/asFeng/Diffuser) 
2. [Gradient-based Intra-attention Pruning on Pre-trained Language Models](https://arxiv.org/abs/2212.07634) [![Publish](https://img.shields.io/badge/2023-ACL-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/airaria/GRAIN) 
3. [Pruning Pre-trained Language Models Without Fine-Tuning](https://aclanthology.org/2023.acl-long.35.pdf) [![Publish](https://img.shields.io/badge/2023-ACL-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/kongds/SMP) 
4. [Pruning Pre-trained Language Models with Principled Importance and Self-regularization](https://aclanthology.org/2023.findings-acl.573/) [![Publish](https://img.shields.io/badge/2023-ACL_Findings-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/drsy/pins) 
5. [Structured Pruning for Efficient Generative Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.692.pdf) [![Publish](https://img.shields.io/badge/2023-ACL_Findings-green)]  
6. [Structural Pruning of Large Language Models via Neural Architecture Search](https://openreview.net/forum?id=SHlZcInS6C) [![Publish](https://img.shields.io/badge/2023-AutoML_Workshop-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/awslabs/syne-tune) 
7. [SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer](https://arxiv.org/abs/2303.17605) [![Publish](https://img.shields.io/badge/2023-CVPR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/sparsevit) 
8. [TorchSparse++: Efficient Point Cloud Engine](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf) [![Publish](https://img.shields.io/badge/2023-CVPR_workshop-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/mit-han-lab/torchsparse) 
9. [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.10512.pdf) [![Publish](https://img.shields.io/badge/2023-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/QingruZhang/AdaLoRA) 
10. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323.pdf) [![Publish](https://img.shields.io/badge/2023-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/gptq) 
11. [Minimum Variance Unbiased N:M Sparsity for the Neural Gradients](https://openreview.net/pdf?id=vuD2xEtxZcj) [![Publish](https://img.shields.io/badge/2023-ICLR-blue)]  
12. [The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers](https://openreview.net/forum?id=TJ2nxciYCk-) [![Publish](https://img.shields.io/badge/2023-ICLR-blue)]  
13. [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://openreview.net/forum?id=wIPIhHd00i) [![Publish](https://img.shields.io/badge/2023-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/FMInference/DejaVu) 
14. [SparseGPT: Massive Language Models Can be Accurately Pruned in one-shot.](https://arxiv.org/pdf/2301.00774.pdf) [![Publish](https://img.shields.io/badge/2023-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/sparsegpt) 
15. [Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) [![Publish](https://img.shields.io/badge/2023-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/yxli2123/LoSparse) 
16. [Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning](https://proceedings.mlsys.org/paper_files/paper/2023/file/4552cedd396a308320209f75f56a5ad5-Paper-mlsys2023.pdf) [![Publish](https://img.shields.io/badge/2023-MLSys-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/SparTA) 
17. [ZipLM: Inference-Aware Structured Pruning of Language Models](https://openreview.net/pdf?id=bPFFPueAxm) [![Publish](https://img.shields.io/badge/2023-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/ZipLM) 
18. [VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores](http://arxiv.org/abs/2310.02065v1) [![Publish](https://img.shields.io/badge/2023-SC-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/UDC-GAC/venom) 
19. [Efficient Memory Management for Large Language Model Serving with PagedAttention](http://arxiv.org/abs/2309.06180v1) [![Publish](https://img.shields.io/badge/2023-SOSP-orange)] ![GitHub Repo stars](https://img.shields.io/github/stars/vllm-project/vllm) 
20. [Efficient Methods for Natural Language Processing: A Survey](https://arxiv.org/abs/2209.00099) [![Publish](https://img.shields.io/badge/2023-TACL-green)]  
21. [SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models](https://arxiv.org/abs/2303.10464) [![Publish](https://img.shields.io/badge/2023-UAI-green)]  
22. [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
23. [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
24. [Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models](http://arxiv.org/abs/2311.04902v2) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/VILA-Lab/GBLM-Pruner) 
25. [Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models](https://arxiv.org/abs/2310.05015) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/Moonlit) 
26. [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers](https://arxiv.org/abs/2305.15805) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
27. [Efficient Guided Generation for Large Language Models](http://arxiv.org/abs/2307.09702v4) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
28. [Fine-Tuning Language Models with Just Forward Passes](http://arxiv.org/abs/2305.17333v3) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/MeZO) 
29. [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
30. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/flash-attention) 
31. [Gradient-Free Structured Pruning with Unlabeled Data](https://arxiv.org/abs/2303.04185) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
32. [H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](http://arxiv.org/abs/2306.14048) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/FMInference/H2O) 
33. [Knowledge-preserving Pruning for Pre-trained Language Models without Retraining](https://arxiv.org/abs/2308.03449) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
34. [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](http://arxiv.org/abs/2312.11514) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
35. [LLM-Pruner: On the Structural Pruning of Large Language Models](http://arxiv.org/abs/2305.11627v3) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/horseee/LLM-Pruner) 
36. [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
37. [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/yxli2123/LoftQ) 
38. [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/OpenGVLab/OmniQuant) 
39. [Post-training Quantization for Neural Networks with Provable Guarantees](https://arxiv.org/pdf/2201.11113.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/YixuanSeanZhou/Quantized_Neural_Nets) 
40. [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](http://arxiv.org/abs/2312.12456v1) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer) 
41. [Pruning Large Language Models via Accuracy Predictor](https://arxiv.org/abs/2309.09507) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
42. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/artidoro/qlora) 
43. [QuIP: Quantization with Incoherence Processing](https://arxiv.org/pdf/2307.13304.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/jerry-chee/QuIP) 
44. [RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/pdf/2304.01089.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/hahnyuan/RPTQ4LLM) 
45. [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://xiamengzhou.github.io/sheared-llama/) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing) 
46. [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/pdf/2306.03078.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/Vahe1994/SpQR) 
47. [Sparse Fine-tuning for Inference Acceleration of Large Language Models](https://arxiv.org/pdf/2310.06927.pdf) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/SparseFinetuning) 
48. [Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](https://arxiv.org/abs/2303.11525) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/CerebrasResearch/Sparse-IFT) 
49. [Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging](https://arxiv.org/abs/2306.16788) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/ZIB-IOL/SMS) 
50. [Ten Lessons We Have Learned in the New Sparseland: A Short Handbook for Sparse Neural Network Researchers](https://arxiv.org/abs/2302.02596) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)]  
51. [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/essential_sparsity) 
52. [Training Transformers with 4-bit Integers](https://arxiv.org/abs//2306.11987) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/xijiu9/Train_Transformers_with_INT4) 
53. [Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering](https://arxiv.org/abs/2304.12102) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/liyucheng09/Selective_Context) 
54. [ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation](https://arxiv.org/abs/2303.08302) [![Publish](https://img.shields.io/badge/2023-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DeepSpeed) 
55. [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) [![Publish](https://img.shields.io/badge/2023-github-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/FasterTransformer) 
</p>
</details>
<details open><summary><b>2022</b></summary> 
<p>

1. [Sparse Progressive Distillation: Resolving Overfitting under Pretrain-and-Finetune Paradigm](https://aclanthology.org/2022.acl-long.16/) [![Publish](https://img.shields.io/badge/2022-ACL-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/shaoyiHusky/SparseProgressiveDistillation) 
2. [TextPruner: A Model Pruning Toolkit for Pre-Trained Language Models](https://arxiv.org/abs/2203.15996) [![Publish](https://img.shields.io/badge/2022-ACL-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/airaria/TextPruner) 
3. [Creating Sparse GPT-3 Models with Iterative Pruning](https://www.cerebras.net/blog/creating-sparse-gpt-3-models-with-iterative-pruning) [![Publish](https://img.shields.io/badge/2022-Blog-green)]  
4. [LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685) [![Publish](https://img.shields.io/badge/2022-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LoRA) 
5. [SPDY: Accurate Pruning with Speedup Guarantees](https://arxiv.org/abs/2201.13096) [![Publish](https://img.shields.io/badge/2022-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/spdy) 
6. [Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation](https://arxiv.org/abs/2209.00606) [![Publish](https://img.shields.io/badge/2022-MICRO-orange)]  
7. [A Fast Post-Training Pruning Framework for Transformers](http://arxiv.org/abs/2204.09656v2) [![Publish](https://img.shields.io/badge/2022-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/WoosukKwon/retraining-free-pruning) 
8. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) [![Publish](https://img.shields.io/badge/2022-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/Dao-AILab/flash-attention) 
9. [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://openreview.net/pdf?id=ksVGCOlOEba) [![Publish](https://img.shields.io/badge/2022-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/IST-DASLab/OBC) 
10. [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://openreview.net/forum?id=f-fVCElZ-G1) [![Publish](https://img.shields.io/badge/2022-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/DeepSpeed) 
11. [Two Sparsities Are Better Than One: Unlocking the Performance Benefits of Sparse-Sparse Networks](https://iopscience.iop.org/article/10.1088/2634-4386/ac7c8a) [![Publish](https://img.shields.io/badge/2022-Neuromorphic_Computing_and_Engineering-green)]  
12. [Transformer Acceleration with Dynamic Sparse Attention](http://arxiv.org/abs/2110.11299v1) [![Publish](https://img.shields.io/badge/2022-TC-green)]  
13. [An Algorithm-Hardware Co-Optimized Framework for Accelerating N:M Sparse Transformers](https://arxiv.org/abs/2208.06118) [![Publish](https://img.shields.io/badge/2022-VLSI-green)]  
14. [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) [![Publish](https://img.shields.io/badge/2022-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/neuralmagic/sparseml) 
</p>
</details>
<details open><summary><b>2021</b></summary> 
<p>

1. [Post-training deep neural network pruning via layer-wise calibration](https://openaccess.thecvf.com/content/ICCV2021W/LPCV/papers/Lazarevich_Post-Training_Deep_Neural_Network_Pruning_via_Layer-Wise_Calibration_ICCVW_2021_paper.pdf) [![Publish](https://img.shields.io/badge/2021-ICCV_workshop-green)]  
2. [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://openreview.net/pdf?id=POWv6hDd9XH) [![Publish](https://img.shields.io/badge/2021-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/yhhhli/BRECQ) 
3. [Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch](https://openreview.net/forum?id=K9bw7vqp_s) [![Publish](https://img.shields.io/badge/2021-ICLR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/aojunzz/NM-sparsity) 
4. [A Greedy Algorithm for Quantizing Neural Networks](https://jmlr.csail.mit.edu/papers/volume22/20-1233/20-1233.pdf) [![Publish](https://img.shields.io/badge/2021-JMLR-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/YixuanSeanZhou/Quantized_Neural_Nets) 
5. [Channel Permutations for N:M Sparsity](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) [![Publish](https://img.shields.io/badge/2021-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/apex) 
6. [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) [![Publish](https://img.shields.io/badge/2021-arXiv-violet)]  
7. [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554) [![Publish](https://img.shields.io/badge/2021-arXiv-violet)]  
</p>
</details>
<details open><summary><b>2020</b></summary> 
<p>

1. [Fast Sparse ConvNets](https://openaccess.thecvf.com/content_CVPR_2020/papers/Elsen_Fast_Sparse_ConvNets_CVPR_2020_paper.pdf) [![Publish](https://img.shields.io/badge/2020-CVPR-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/fastconvnets/cvpr2020) 
2. [Inducing and Exploiting Activation Sparsity for Fast Neural Network Inference](http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf) [![Publish](https://img.shields.io/badge/2020-ICML-blue)]  
3. [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) [![Publish](https://img.shields.io/badge/2020-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/block_movement_pruning) 
4. [GPU Kernels for Block-Sparse Weights](https://cdn.openai.com/blocksparse/blocksparsepaper.pdf) [![Publish](https://img.shields.io/badge/2020-arXiv-violet)] ![GitHub Repo stars](https://img.shields.io/github/stars/openai/blocksparse) 
</p>
</details>
<details open><summary><b>2019</b></summary> 
<p>

1. [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129) [![Publish](https://img.shields.io/badge/2019-ICML-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/ucbrise/actnn) 
</p>
</details>
<details open><summary><b>2018</b></summary> 
<p>

1. [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294) [![Publish](https://img.shields.io/badge/2018-ECCV-green)] ![GitHub Repo stars](https://img.shields.io/github/stars/bzantium/pytorch-admm-pruning) 
</p>
</details>
<details open><summary><b>2017</b></summary> 
<p>

1. [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/pdf/1607.04381.pdf) [![Publish](https://img.shields.io/badge/2017-ICLR-blue)]  
2. [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/pdf/1705.07565.pdf) [![Publish](https://img.shields.io/badge/2017-NeurIPS-blue)] ![GitHub Repo stars](https://img.shields.io/github/stars/csyhhu/L-OBS) 
</p>
</details>
<details open><summary><b>2016</b></summary> 
<p>

1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf) [![Publish](https://img.shields.io/badge/2016-ICLR-blue)]  
</p>
</details>
<details open><summary><b>1993</b></summary> 
<p>

1. [Optimal Brain Surgeon and general network pruning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=298572&tag=1) [![Publish](https://img.shields.io/badge/1993--green)]  
</p>
</details>
<details open><summary><b>1989</b></summary> 
<p>

1. [Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [![Publish](https://img.shields.io/badge/1989-NeurIPS-blue)]  
</p>
</details>
</p>
</details>

## References

### Hot
1. https://github.com/horseee/Awesome-Efficient-LLM
2. https://github.com/DefTruth/Awesome-Diffusion-Inference
3. https://github.com/DefTruth/Awesome-LLM-Inference
4. https://github.com/AmberLJC/LLMSys-PaperList
5. https://github.com/Hannibal046/Awesome-LLM
6. https://github.com/AmadeusChan/Awesome-LLM-System-Papers
7. https://github.com/KnowingNothing/compiler-and-arch
8. https://papercopilot.com/paper-list
9. https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
10. https://github.com/October2001/Awesome-KV-Cache-Compression

### Cold
1. https://github.com/he-y/Awesome-Pruning
2. https://github.com/htqin/awesome-model-quantization
3. https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression
4. https://github.com/AojunZhou/Efficient-Deep-Learning
5. https://github.com/chester256/Model-Compression-Papers

[:arrow_up: Back to top](#efficientpaper)
