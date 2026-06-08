# EfficientPaper 统一更新计划

## 状态概览（2026-06-04 创建）

- **总论文数**: 474
- **按年份分布**: 2026: 71 | 2025: 193 | 2024: 119 | 2023: 57 | 2022: 15 | 2021: 7 | 更早: 12
- **Note 状态**: 缺失 75 | 极短 <500c: 22 | 部分 500-3000c: 339 | 完整 >=3000c: 37
- **Baseline 状态**: 无/None: 344
- **Code URL 缺失**: 186
- **Keyword 问题**: 无

## 更新策略

1. **按年份从新到旧**：2026 → 2025 → 2024 → 更早
2. **每篇论文**：
   - 检查 note 完整性，若已有简要 note 则在后追加，不删除已有内容
   - 检查 prototxt：baseline methods（`year/abbr` 格式，只引用已有 paper）、keywords（合法 enum）、code URL
   - 不修改 update_time 时间戳
3. **每篇论文完成后**：运行 `./refresh_and_upload.sh` 验证，有报错及时修正
4. **每完成一批**：更新本计划

## 处理进度

### 2026 年（71 篇，44 篇需更新 note）

**需更新 note 的论文（note < 3000c）**：

- [x] AdaSplash-2 (1701c/1s) → 5,762c, complete Chinese note with abstract, background, method (histogram initialization, hybrid solver, GPU kernel), results (2× speedup at high sparsity, long-context gains), advantages/limitations
- [x] AttentionResiduals (2134c/1s) → 4,982c, complete Chinese note with abstract, background, method (AttnRes, Block AttnRes, cross-stage caching, two-phase computation), results (scaling law, 48B model, downstream gains), advantages/limitations
- [x] AutoOverlap (1327c/1s) → 8,152c, complete Chinese note with abstract, background, method (chunk abstraction, adaptive backend/scheduling/chunk size), results (1.3× avg, 4.7× max), advantages/limitations
- [x] CacheFlow (2381c/2s) → 6,118c, complete Chinese note with abstract, background, method (3D parallelism, two-pointer scheduler, multi-GPU parallelism, batch-aware scheduling), results (10%-62% TTFT reduction, 88% GPU + 78% I/O utilization), advantages/limitations
- [x] Double-P (1406c/1s) → 7,734c, complete Chinese note with abstract, background, method (hierarchical top-p, cluster-level + token-level), results (1.8× reduction, 1.3× speedup), advantages/limitations
- [x] DualMap (2303c/1s) → 15,779c, complete Chinese note with abstract, background, method (SLO-aware routing, hot-spot rebalancing, lightweight dual-hash ring expansion), results (SLO compliance, throughput improvement), advantages/limitations
- [x] DualPath (1615c/1s) → 12,321c, complete Chinese note with abstract, background, method (dual-path storage bandwidth bottleneck breaking), results, advantages/limitations
- [x] Engram (43c/1s) → 11,224c, complete Chinese note with abstract, background, method, results, advantages, limitations, and relation to EfficientPaper
- [x] FSR (1535c/1s) → 6,880c, complete Chinese note with abstract, background, method (Focus-Scan-Refine, dynamic allocation), results (outperforms SOTA pruning), advantages/limitations
- [x] FastKVzip (1289c/1s) → 8,303c, complete Chinese note with abstract, background, method (gated KV eviction, sink-attention, training), results (70% eviction, near-lossless), advantages/limitations
- [x] FlashAttention-4 (1716c/1s) → 13,738c, complete Chinese note with abstract, background, method (forward/backward 4 sections, scheduling strategy), results (forward/backward/throughput/compilation/baseline comparison), advantages/limitations
- [x] FlashOverlap (1629c/1s) → 12,759c, complete Chinese note with abstract, background, method (signal mechanism, reordering, design space, prediction search, implementation), results (hardware benchmark, core performance), advantages/limitations
- [x] FlashPrefill (1462c/1s) → 7,128c, complete Chinese note with abstract, background, method (instantaneous pattern discovery, max-based dynamic thresholding), results (27.78× speedup, 1.71× at 4K), advantages/limitations
- [x] GatedNorm (1851c/1s) → 10,077c, complete Chinese note with abstract, background, method (5 experiments, 2B dense + MoE large-scale + quantization), results (outperforms SOTA), advantages/limitations
- [x] HERMES (1412c/1s) → 7,943c, complete Chinese note with abstract, background, method (hierarchical KV cache management, cross-layer smoothing, position re-indexing), results (10× TTFT, 68% token reduction, 11.4% gain), advantages/limitations
- [x] HISA (1890c/1s) → 12,468c, complete Chinese note with abstract, background, method (two-stage hierarchical search, block coarse-filtering + token refinement, complexity analysis), results (kernel speedup up to 3.75×, NIAH, LongBench), advantages/limitations
- [x] HN5FDNZ3 (728c/1s) → 7,755c, complete Chinese note with abstract, background, method (L1/L2 cache analysis, L2 modeling, sawtooth reorder), results (CUDA/CuTile), advantages/limitations
- [x] HySparse (1693c/1s) → 14,616c, complete Chinese note with abstract, background, method (5 sub-methods, training config, ablations), results (general benchmarks + long-context RULER), advantages/limitations
- [x] IBW4TYDG (1507c/1s) → 7,140c, complete Chinese note with abstract, background, method (systematic QAT study, KD, PTQ init, RL, data strategy), results (44.53% over GPTQ), advantages/limitations
- [x] IndexCache (2259c/1s) → 9,649c, complete Chinese note with abstract, background, method (greedy search, multi-layer distillation, training-agnostic/aware), results (end-to-end speedup, GLM-5 extension), advantages/limitations
- [x] K-Search (1744c/1s) → 10,186c, complete Chinese note with abstract, background, method (co-evolution world model, three-phase iteration, search tree), results (FlashInfer kernel comparison, GPUMode TriMul), advantages/limitations
- [x] KV-CAT (2584c/2s) → 11,286c, complete Chinese note with abstract, background, method (KV cache compression-aware training, router sparsification, theory), results (+6.4 retrieval, +39% long-context QA, 3.21× compression, 5× optimization), advantages/limitations
- [x] KVTC (1616c/1s) → 11,066c, complete Chinese note with abstract, background, method (PCA decorrelation, dynamic programming quantization, entropy coding, sliding window protection, three working modes), results (general/reasoning/multi-GPU/latency), advantages/limitations
- [x] KVzap (821c/1s) → 9,414c, complete Chinese note with abstract, background, method (KVzip/KVzip+/KVzap), results, ablations, advantages/limitations
- [x] L3 (1432c/1s) → 6,171c, complete Chinese note with abstract, background, method (L³ layer, static token routing, LZW allocation), results (outperforms dense+MoE), advantages/limitations
- [x] LycheeDecode (1711c/1s) → 11,943c, complete Chinese note with abstract, background, method (mixed head framework, HardKuma distribution, custom kernels, training setup), results (LongBench, AIME24, RULER, efficiency, ablations), advantages/limitations
- [x] MixServe (1968c/1s) → 10,755c, complete Chinese note with abstract, background, method (auto analyzer, hybrid TP-EP partitioner, fused AR-A2A communication, performance modeling), results (TTFT/ITL/throughput, ablations), advantages/limitations
- [x] PAT (1689c/1s) → 12,029c, complete Chinese note with abstract, background, method (pack-forward-merge paradigm, Pack scheduler, multi-tile kernel, multi-stream forwarding, KV split, output merge), results (kernel performance, end-to-end, ablations, overhead), advantages/limitations
- [x] PackCache (1867c/1s) → 12,605c, complete Chinese note with abstract, background, method (KV-Cache bottleneck, visual token, attention pattern, three-stage mechanism, spatial-preserving position embedding), results (quality/efficiency tables, ablations), advantages/limitations
- [x] PrfaaS (2388c/1s) → 11,695c, complete Chinese note with abstract, background, method (prefill acceleration), results, advantages/limitations
- [x] Prism (1708c/1s) → 11,937c, complete Chinese note with abstract, background, method (spectral decay theory, energy analysis, dual-frequency branch design, temperature calibration), results (language modeling, long-context, retrieval, video understanding, efficiency, ablations), advantages/limitations
- [x] SDFP (1463c/1s) → 6,691c, complete Chinese note with abstract, background, method (FIT-based layer pruning, speculative decoding), results (1.32×-1.5× speedup), advantages/limitations
- [x] SPEED (2614c/2s) → 11,898c, complete Chinese note with abstract, background, method (token visibility rules, cost model, BoS anchor, hierarchical truncation diagnosis), results (truncation scan, BoS analysis, task dependency, comparison with SwiftKV/POP, LoRA adaptation, training efficiency, SelfOnly ablation), advantages/limitations
- [x] SparKV (1412c/1s) → 7,461c, complete Chinese note with abstract, background, method (KV chunk scheduler, overhead model, runtime controller), results (1.3×-5.1× TTFT, 1.5×-3.3× energy), advantages/limitations
- [x] SparVAR (2054c/1s) → 12,609c, complete Chinese note with abstract, background, method (CS4A, CSLA, cross-scale sparse index mapping), results (GenEval/DPG-Bench/HPSv2.1/ImageReward/PSNR/SSIM/LPIPS, human preference, ablations), advantages/limitations
- [x] SparseForcing (1975c/1s) → 12,795c, complete Chinese note with abstract, background, method (PBSA, block compression, coarse scoring, Top-K selection, GPU kernel, training strategy), results (short video, long video 20s/1min, ablations, kernel performance), advantages/limitations
- [x] SparseForge (2320c/2s) → 12,433c, complete Chinese note with abstract, background, method (dual-track retraining loop, Hessian-aware soft mask update, progressive quenching), results (main results, cross-model, ablations, block-16 extension, inference acceleration), advantages/limitations
- [x] Tactic (1735c/1s) → 11,868c, complete Chinese note with abstract, background, method (technical details), results, advantages/limitations
- [x] TurboQuant (2094c/1s) → 13,394c, complete Chinese note with abstract, background, method (MSE-optimal TurboQuant, inner-product-optimal TurboQuant, information-theoretic lower bound), results (theoretical verification, NIAH, LongBench, neighbor search), advantages/limitations
- [x] UniPrefill (2760c/2s) → 15,264c, complete Chinese note with abstract, background, method (token importance estimation, top-p selection, sparsity propagation, fused kernel & vLLM integration), results (RULER, vLLM throughput, ablations), advantages/limitations
- [x] VQKV (1230c/1s) → 8,010c, complete Chinese note with abstract, background, method (RSimVQ, prefill/decode), results, advantages/limitations
- [x] X3NUE78O (1110c/1s) → 7,291c, complete Chinese note with abstract, background, method (per-channel INT8 quantization, 4 CUDA kernels), results (1,694× speedup), advantages/limitations
- [x] XAX01V4E (2369c/1s) → 8,230c, complete Chinese note with abstract, background, method (experimental setup, four cache conditions, evaluation metrics, ablation study), results (core findings, best cache strategy, ablation conclusions), advantages/limitations
- [x] ZipServ (1789c/1s) → 14,955c, complete Chinese note with abstract, background, method (TCA-TBE encoding, ZipGEMM kernel, stage-aware strategy, implementation details), results (kernel-level, end-to-end, memory savings, overhead analysis), advantages/limitations

**note 已完整（>=3000c）的论文**：27 篇（仍需检查 baseline 方法）

### 2025 年（192 篇）- 待处理
### 2024 年（119 篇）- 待处理
### 2023 年及更早（81 篇）- 待处理

## 已完成的论文

| 日期 | 年份 | Abbr | 变更摘要 |
|------|------|------|----------|
| 2026-06-04 | 2026 | FSR | 补充完整中文 note（6,880c），含背景、方法（Focus-Scan-Refine、动态分配）、实验（超越 SOTA 剪枝）、优点与局限 |
| 2026-06-04 | 2026 | IBW4TYDG | 补充完整中文 note（7,140c），含背景、方法（系统 QAT 研究、KD、PTQ 初始化、RL、数据策略）、实验（44.53% over GPTQ）、优点与局限 |
| 2026-06-04 | 2026 | SDFP | 补充完整中文 note（6,691c），含背景、方法（FIT 层剪枝、推测解码）、实验（1.32×-1.5× 加速）、优点与局限 |
| 2026-06-04 | 2026 | FlashPrefill | 补充完整中文 note（7,128c），含背景、方法（瞬时模式发现、基于 Max 的动态阈值）、实验（27.78× 加速、1.71× at 4K）、优点与局限 |
| 2026-06-04 | 2026 | L3 | 补充完整中文 note（6,171c），含背景、方法（L³ 层、静态 token 路由、LZW 分配）、实验（超越稠密+MoE）、优点与局限 |
| 2026-06-04 | 2026 | SparKV | 补充完整中文 note（7,461c），含背景、方法（KV 块调度器、开销模型、运行时控制器）、实验（1.3×-5.1× TTFT、1.5×-3.3× 能耗）、优点与局限 |
| 2026-06-04 | 2026 | HERMES | 补充完整中文 note（7,943c），含背景、方法（分层 KV 缓存管理、跨层记忆平滑、位置重索引）、实验（10× TTFT、68% token 减少、11.4% 提升）、优点与局限 |
| 2026-06-04 | 2026 | Double-P | 补充完整中文 note（7,734c），含背景、方法（分层 top-p、集群级+token 级）、实验（1.8× 减少、1.3× 加速）、优点与局限 |
| 2026-06-04 | 2026 | AutoOverlap | 补充完整中文 note（8,152c），含背景、方法（chunk 抽象、自适应后端/调度/块大小）、实验（1.3× 平均、4.7× 最高）、优点与局限 |
| 2026-06-04 | 2026 | FastKVzip | 补充完整中文 note（8,303c），含背景、方法（门控 KV 剪枝、sink-attention、训练）、实验（70% 剪枝、近无损性能）、优点与局限 |
| 2026-06-04 | 2026 | X3NUE78O | 补充完整中文 note（7,291c），含背景、方法（每通道 INT8 量化、4 种 CUDA 内核）、实验（1,694× 加速）、优点与局限 |
| 2026-06-04 | 2026 | VQKV | 补充完整中文 note（8,010c），含背景、方法（RSimVQ、预填充/解码）、实验、优点与局限 |
| 2026-06-04 | 2026 | HN5FDNZ3 | 补充完整中文 note（7,755c），含背景、方法（L1/L2 缓存分析、L2 建模、锯齿波前重排序）、实验（CUDA/CuTile）、优点与局限 |
| 2026-06-04 | 2026 | KVzap | 补充完整中文 note（9,414c），含背景、方法（KVzip/KVzip+/KVzap）、实验、结果、消融、优点与局限 |
| 2026-06-04 | 2026 | Engram | 补充完整中文 note（11,224c），含背景、方法、实验、结果、优点与局限 |
| 2026-06-04 | 2026 | RTP-LLM | 添加 code URL: https://github.com/alibaba/rtp-llm |
| 2026-06-04 | 2024 | Mooncake | 补充完整中文 note（含背景、方法、实验、结果、优点与局限） |
| 2026-06-04 | 2025 | FastTree | 新增论文：完整 prototxt 元数据 + 中文 note（6,443c），含背景、方法（树结构分组、贪心启发式、GPU kernel）、实验（5.1×/4.2×/10.6×/2.1× 加速，2.2× 端到端）、优点与局限 |
| 2026-06-04 | 2026 | AdaSplash-2 | 补充完整中文 note（5,762c），含背景、方法（直方图初始化、混合求解器、GPU kernel）、实验（2× 加速、长上下文收益）、优点与局限 |
| 2026-06-04 | 2026 | AttentionResiduals | 补充完整中文 note（4,982c），含背景、方法（AttnRes、Block AttnRes、跨阶段缓存、两阶段计算）、实验（scaling law、48B 模型、下游任务）、优点与局限 |
| 2026-06-04 | 2026 | CacheFlow | 补充完整中文 note（6,118c），含背景、方法（3D 并行、双指针调度器、multi-GPU 并行、batch 感知）、实验（10%-62% TTFT 降低、88% GPU + 78% I/O 利用率）、优点与局限 |
| 2026-06-04 | 2026 | DualMap | 补充完整中文 note（15,779c），含背景、方法（SLO 感知路由、热点感知重平衡、轻量级双哈希环扩展）、实验（SLO 合规、吞吐量提升）、优点与局限 |
| 2026-06-04 | 2026 | DualPath | 补充完整中文 note（12,321c），含背景、方法（双路径存储带宽瓶颈突破）、实验、优点与局限 |
| 2026-06-04 | 2026 | FlashAttention-4 | 补充完整中文 note（13,738c），含背景、方法（前向/反向 4 节、调度策略）、实验（前向/反向/吞吐量/编译/基线对比）、优点与局限 |
| 2026-06-04 | 2026 | FlashOverlap | 补充完整中文 note（12,759c），含背景、方法（信号机制、重排、设计空间、预测搜索、实现细节）、实验（硬件基准、核心性能）、优点与局限 |
| 2026-06-04 | 2026 | GatedNorm | 补充完整中文 note（10,077c），含背景、方法（5 组实验、2B 密集 + MoE 大规模 + 量化）、实验（超越 SOTA）、优点与局限 |
| 2026-06-04 | 2026 | HySparse | 补充完整中文 note（14,616c），含背景、方法（5 子方法、训练配置、消融）、实验（通用基准 + 长上下文 RULER）、优点与局限 |
| 2026-06-04 | 2026 | HISA | 补充完整中文 note（12,468c），含背景、方法（两阶段分层搜索、块级粗过滤 + 令牌级精炼、复杂度分析）、实验（内核加速最高 3.75×、NIAH、LongBench）、优点与局限 |
| 2026-06-04 | 2026 | IndexCache | 补充完整中文 note（9,649c），含背景、方法（贪心搜索、多层蒸馏、训练无关/感知）、实验（端到端加速、GLM-5 扩展）、优点与局限 |
| 2026-06-04 | 2026 | K-Search | 补充完整中文 note（10,186c），含背景、方法（协同进化世界模型、三阶段迭代、搜索树）、实验（FlashInfer kernel 对比、GPUMode TriMul）、优点与局限 |
| 2026-06-04 | 2026 | KV-CAT | 补充完整中文 note（11,286c），含背景、方法（KV 缓存压缩感知训练、路由器稀疏化、理论）、实验（+6.4 检索、+39% 长上下文 QA、3.21× 压缩、5× 优化）、优点与局限 |
| 2026-06-04 | 2026 | KVTC | 补充完整中文 note（11,066c），含背景、方法（PCA 解相关、动态规划量化、熵编码、滑动窗口保护、三种工作模式）、实验（通用/推理/多 GPU/延迟）、优点与局限 |
| 2026-06-04 | 2026 | LycheeDecode | 补充完整中文 note（11,943c），含背景、方法（混合头框架、HardKuma 分布、自定义内核、训练设置）、实验（LongBench、AIME24、RULER、效率、消融）、优点与局限 |
| 2026-06-04 | 2026 | MixServe | 补充完整中文 note（10,755c），含背景、方法（自动分析器、混合 TP-EP 分区器、融合 AR-A2A 通信、性能建模）、实验（TTFT/ITL/吞吐量、消融）、优点与局限 |
| 2026-06-04 | 2026 | PAT | 补充完整中文 note（12,029c），含背景、方法（pack-forward-merge 范式、Pack 调度器、多 tile 内核、多流转发、KV 分割、输出合并）、实验（内核性能、端到端、消融、开销）、优点与局限 |
| 2026-06-04 | 2026 | PackCache | 补充完整中文 note（12,605c），含背景、方法（KV-Cache 瓶颈、视觉 token、注意力模式、三阶段机制、空间保持位置嵌入）、实验（质量/效率表格、消融）、优点与局限 |
| 2026-06-04 | 2026 | PrfaaS | 补充完整中文 note（11,695c），含背景、方法（prefill 加速）、实验、优点与局限 |
| 2026-06-04 | 2026 | Prism | 补充完整中文 note（11,937c），含背景、方法（频谱衰减理论、能量分析、双频分支设计、温度校准）、实验（语言建模、长上下文、检索、视频理解、效率、消融）、优点与局限 |
| 2026-06-04 | 2026 | SPEED | 补充完整中文 note（11,898c），含背景、方法（token 可见性规则、成本模型、BoS 锚点、层次截断诊断）、实验（截断扫描、BoS 分析、任务依赖性、SwiftKV/POP 对比、LoRA 适配、训练效率、SelfOnly 消融）、优点与局限 |
| 2026-06-04 | 2026 | SparVAR | 补充完整中文 note（12,609c），含背景、方法（CS4A、CSLA、跨尺度稀疏索引映射）、实验（GenEval/DPG-Bench/HPSv2.1/ImageReward/PSNR/SSIM/LPIPS、人类偏好、消融）、优点与局限 |
| 2026-06-04 | 2026 | SparseForcing | 补充完整中文 note（12,795c），含背景、方法（PBSA、块化压缩、粗略评分、Top-K 选择、GPU 内核、训练策略）、实验（短视频、长视频 20s/1min、消融、内核性能）、优点与局限 |
| 2026-06-04 | 2026 | SparseForge | 补充完整中文 note（12,433c），含背景、方法（双轨重训练循环、Hessian 感知软掩码更新、渐进淬火）、实验（主结果、跨模型、消融、block-16 扩展、推理加速）、优点与局限 |
| 2026-06-04 | 2026 | Tactic | 补充完整中文 note（11,868c），含背景、方法（技术细节）、实验、优点与局限 |
| 2026-06-04 | 2026 | TurboQuant | 补充完整中文 note（13,394c），含背景、方法（MSE 最优 TurboQuant、内积最优 TurboQuant、信息论下界）、实验（理论验证、NIAH、LongBench、近邻搜索）、优点与局限 |
| 2026-06-04 | 2026 | UniPrefill | 补充完整中文 note（15,264c），含背景、方法（token 重要性估算、top-p 选择、稀疏性传播、融合内核与 vLLM 集成）、实验（RULER、vLLM 吞吐量、消融）、优点与局限 |
| 2026-06-04 | 2026 | XAX01V4E | 补充完整中文 note（8,230c），含背景、方法（实验设置、四种缓存条件、评估指标、消融研究）、实验（核心发现、最佳缓存策略、消融结论）、优点与局限 |
| 2026-06-05 | 2024 | PowerInfer-2 | 补充完整中文 note（14,939c），含背景、方法（神经元集群抽象、多态神经元引擎、内存神经元缓存、灵活神经元加载、集群级流水线、离线规划器）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SlimGPT | 补充完整中文 note（10,487c），含背景、方法（基于 OBS 的低成本快速结构化剪枝、批量贪心剪枝、递增剪枝比例）、实验结果（LLaMA-7B/13B/30B）、优点与局限 |
| 2026-06-05 | 2023 | LLM-Pruner | 补充完整中文 note（12,945c），含背景、方法（3 阶段：Discovery、Estimation、Recovery、Group Types）、实验结果（LLaMA-7B/13B、Vicuna-7B、ChatGLM-6B）、优点与局限 |
| 2026-06-05 | 2024 | LightningAttention-2 | 补充完整中文 note（12,019c），含背景、方法（前向/反向传播、tiling、块内/块间分离、硬件优化）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | 068ZPAME | 补充完整中文 note（17,992c），含背景、方法（MoE 推理优化综述：模型层、系统层、硬件层）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | KIVI | 补充完整中文 note（10,408c），含背景、方法（非对称 2bit KV Cache 量化）、实验结果（2.6× 峰值内存减少、2.35×~3.47× 吞吐提升）、优点与局限 |
| 2026-06-05 | 2024 | APEX | 补充完整中文 note（14,267c），含背景、方法（动态感知模拟器、DP+PP+TP+EP）、实验结果（3.37× 加速、45% 能耗降低）、优点与局限 |
| 2026-06-05 | 2024 | CATS | 补充完整中文 note（10,643c），含背景、方法（稀疏激活剪枝）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | LLM_in_a_flash | 补充完整中文 note（11,063c），含背景、方法（低秩预测器、选择性持久化、滑动窗口、行列捆绑）、实验结果（I/O 延迟分析表、端到端延迟表）、优点与局限 |
| 2026-06-05 | 2024 | LightningAttention-2 | 补充完整中文 note（12,019c），含背景、方法（前向/反向传播、tiling、块内/块间分离、硬件优化）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | 068ZPAME | 补充完整中文 note（17,992c），含背景、方法（MoE 推理优化综述：模型层、系统层、硬件层）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | KIVI | 补充完整中文 note（10,408c），含背景、方法（非对称 2bit KV Cache 量化）、实验结果（2.6× 峰值内存减少、2.35×~3.47× 吞吐提升）、优点与局限 |
| 2026-06-05 | 2024 | CachedAttention | 补充完整中文 note（11,394c），含背景、方法（4 个子技术）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | ProSparse | 补充完整中文 note（10,554c），含背景、方法（ProSparse 稀疏化）、实验结果、优点与局限 |
| 2026-06-05 | 2022 | ComplementarySparsity | 补充完整中文 note（15,568c），含背景、方法（互补稀疏性、FPGA 实现）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | Minitron | 补充完整中文 note（9,720c），含背景、方法（深度/宽度剪枝、重要性估计、教师校正、知识蒸馏）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SeerAttention | 补充完整中文 note（14,090c），含背景、方法（稀疏注意力预测）、实验结果、优点与局限 |
| 2026-06-05 | 2022 | CoCoNet | 补充完整中文 note（10,824c），含背景、方法（DSL、张量布局、4 种变换、自动调优器、代码生成器）、实验结果（1.2×–2.0× 数据并行）、优点与局限 |
| 2026-06-05 | 2023 | MeZO | 补充完整中文 note（11,046c），含背景、方法（内存高效零阶优化）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | GPUSQ-ViT | 补充完整中文 note（11,235c），含背景、方法（GPU 可实现的 ViT 量化）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | CacheGen | 补充完整中文 note（11,646c），含背景、方法（KV 缓存压缩与生成）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | DeepSeekMoE | 补充完整中文 note（13,838c），含背景、方法（细粒度专家分割、共享专家隔离、负载均衡、消融结果）、实验结果（2B/16B/145B）、优点与局限 |
| 2026-06-05 | 2024 | streaming-llm | 补充完整中文 note（13,063c），含背景、方法（注意力汇聚现象、滚动 KV 缓存、预训练 Sink Token、位置编码处理）、实验结果（400 万 token、22.2× 加速）、优点与局限 |
| 2026-06-05 | 2024 | FLUX | 补充完整中文 note（11,280c），含背景、方法（算法、优化：tile swizzling、reduce、AllGather、auto-tuning）、实验结果（A100/H800）、优点与局限 |
| 2026-06-05 | 2024 | GEAR | 补充完整中文 note（10,092c），含背景、方法（量化、低秩近似、稀疏矩阵）、实验结果（5.07× 吞吐提升、2.39× 内存减少）、优点与局限 |
| 2026-06-05 | 2024 | SMAT | 补充完整中文 note（13,428c），含背景、方法（稀疏插值专家、超网络选择、密集教师知识蒸馏、拉格朗日约束控制稀疏度）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | RecycledAttention | 补充完整中文 note（10,815c），含背景、方法（双 KV 缓存 Cf/Cr、固定/动态调度策略）、实验结果（2× 优于 StreamingLLM/H2O）、优点与局限 |
| 2026-06-05 | 2024 | InfLLM | 补充完整中文 note（12,640c），含背景、方法（训练无关内存方法、块级内存单元、动态查找）、实验结果（57.7% ∞-Bench、1024K 令牌、26GB VRAM）、优点与局限 |
| 2026-06-05 | 2024 | DoubleSparsity | 补充完整中文 note（11,061c），含背景、方法（离线校准、标签缓存、算法流程、Offload、复杂度分析）、实验结果（准确性评估、加速评估、消融研究）、优点与局限 |
| 2026-06-05 | 2024 | RazorAttention | 补充完整中文 note（11,649c），含背景、方法（ALiBi/RoPE 模型、补偿 token、检索头识别）、实验结果（LongBench、Needle In A Haystack）、优点与局限 |
| 2026-06-05 | 2024 | LightningAttention | 补充完整中文 note（12,423c），含背景、方法（内块常规注意力 + 外块线性注意力核技巧、TNL 架构、LRPE-d、GLA 门控）、实验结果（11× 推理吞吐提升）、优点与局限 |
| 2026-06-05 | 2024 | Vidur | 补充完整中文 note（12,287c），含背景、方法（Profiler、Runtime Estimator、Hierarchical Scheduler、Vidur-Bench、Vidur-Search）、实验结果（保真度分析、What-if 分析、Pareto 前沿分析）、优点与局限 |
| 2026-06-05 | 2024 | SEA | 补充完整中文 note（10,543c），含背景、方法（SEA 注意力、CNN 解码器、分组 top-k 选择、FlatCSR）、实验结果（Wikitext2、GLUE、效率分析）、优点与局限 |
| 2026-06-05 | 2024 | XGrammar | 补充完整中文 note（12,199c），含背景、方法（词汇表划分、自适应缓存、100× CFG 加速、80× 端到端加速）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | GBLM-Pruner | 补充完整中文 note（13,384c），含背景、方法（理论推导、剪枝指标公式、算法伪代码）、实验结果（困惑度表格、零样本评估、消融实验）、优点与局限 |
| 2026-06-05 | 2024 | SN1PK7EK | 补充完整中文 note（10,981c），含背景、方法（RGE 公式、6 种 ZO 方法、块级、混合、稀疏）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | CHESS | 补充完整中文 note（9,575c），含背景、方法（通道级阈值、选择性稀疏化）、实验结果（1.27× 加速）、优点与局限 |
| 2026-06-05 | 2024 | InfiniteBench | 补充完整中文 note（9,658c），含背景、方法（12 个任务的长上下文基准测试）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | PrefixQuant | 补充完整中文 note（12,744c），含背景、方法（异常值 token 定义、前缀 token 选择、KV 缓存计算公式、块级微调）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SliceGPT | 补充完整中文 note（13,250c），含背景、方法（计算不变性、PCA 正交变换、权重矩阵行/列切片）、实验结果（25% 参数移除保持 99% 零样本性能）、优点与局限 |
| 2026-06-05 | 2024 | SparseLLM | 补充完整中文 note（13,535c），含背景、方法（辅助变量公式、交替优化、闭式解、FFN-first 剪枝策略）、实验结果（OPT 125m-66b、LLaMA-2 7b/13b）、优点与局限 |
| 2026-06-05 | 2024 | MaskLLM | 补充完整中文 note（9,319c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2024 | DistAttention | 补充完整中文 note（13,393c），含背景、方法（DistAttention 机制、集群级调度、gManager/rManager 架构）、实验结果（32 A100 GPU、1.35-3.4× 吞吐提升）、优点与局限 |
| 2026-06-05 | 2024 | Pruner-Zero | 补充完整中文 note（14,210c），含背景、方法（遗传编程自动剪枝度量搜索、OOS 策略、||W|×|W||×σ(|G|) 度量）、实验结果（优于 SparseGPT 和 Wanda）、优点与局限 |
| 2026-06-05 | 2024 | TinyTrain | 补充完整中文 note（11,633c），含背景、方法（多目标准则、动态通道选择、FSL 预训练）、实验结果（精度、内存/计算、延迟/能耗）、优点与局限 |
| 2026-06-05 | 2024 | PowerInfer-2 | 补充完整中文 note（14,939c），含背景、方法（神经元集群抽象、多态神经元引擎、内存神经元缓存、灵活神经元加载、集群级流水线、离线规划器）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SlimGPT | 补充完整中文 note（10,487c），含背景、方法（基于 OBS 的低成本快速结构化剪枝、批量贪心剪枝、递增剪枝比例）、实验结果（LLaMA-7B/13B/30B）、优点与局限 |
| 2026-06-05 | 2023 | LLM-Pruner | 补充完整中文 note（12,945c），含背景、方法（3 阶段：Discovery、Estimation、Recovery、Group Types）、实验结果（LLaMA-7B/13B、Vicuna-7B、ChatGLM-6B）、优点与局限 |
| 2026-06-05 | 2024 | SageAttention2 | 补充完整中文 note（14,849c），含背景、方法（per-thread INT4 量化、Q/K 异常值平滑、FP8 两级累加）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | CLA | 补充完整中文 note（12,047c），含背景、方法（跨层注意力减少 KV 缓存大小）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | AdaKV | 补充完整中文 note（9,644c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2024 | Sparse-IFT | 补充完整中文 note（10,033c），含背景、方法（Sparse Wide、Sparse Parallel、Sparse Factorized、Sparse Doped）、实验结果（ImageNet、CIFAR-100、MS COCO、CityScapes、GPT-3 Small）、优点与局限 |
| 2026-06-05 | 2024 | SPP | 补充完整中文 note（13,956c），含背景、方法（SPP 框架、参数插入、内存优化）、实验结果（零样本评估、5-shot MMLU、消融研究）、优点与局限 |
| 2026-06-05 | 2024 | TurboSparse | 补充完整中文 note（10,420c），含背景、方法（dReLU 公式、小模型验证、稀疏度-性能分析、MoE 稀疏性）、实验结果（下游任务表格、CPU/混合/移动端推理加速）、优点与局限 |
| 2026-06-05 | 2024 | ChunkAttention | 补充完整中文 note（14,089c），含背景、方法（分块注意力优化）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | ULY1AZGY | 补充完整中文 note（10,810c），含背景、方法（高稀疏度基础 Llama 模型、高效预训练与部署）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | k_pruning | 补充完整中文 note（11,397c），含背景、方法（k 剪枝、公式与算法流程）、实验结果（准确性、推理速度、消融实验）、优点与局限 |
| 2026-06-05 | 2022 | fisherpruning | 补充完整中文 note（10,622c），含背景、方法（3 阶段流水线：Mask Search、Mask Rearrangement、Mask Tuning）、实验结果（BERT/DistilBERT on GLUE/SQuAD）、优点与局限 |
| 2026-06-05 | 2024 | Q-Sparse | 补充完整中文 note（9,109c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2024 | ShadowLLM | 补充完整中文 note（10,142c），含背景、方法（plainact 梯度感知剪枝、统一预测器）、实验结果（15%+ 精度提升、20% 加速）、优点与局限 |
| 2026-06-05 | 2022 | DSA | 补充完整中文 note（9,829c），含背景、方法（动态稀疏注意力、硬件协同设计）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | HYPL7G37 | 补充完整中文 note（12,518c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2023 | PowerInfer | 补充完整中文 note（13,774c），含背景、方法（GPU-CPU 混合推理、幂律神经元激活分布）、实验结果（11.69× 加速）、优点与局限 |
| 2026-06-05 | 2017 | Transformer | 补充完整中文 note（15,588c），含背景、方法（Scaled Dot-Product Attention、Multi-Head Attention、FFN、位置编码、训练细节）、实验结果（英德/英法翻译 BLEU、消融实验）、优点与局限 |
| 2026-06-05 | 2024 | Wanda | 补充完整中文 note（11,724c），含背景、方法（简单有效的剪枝方法）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SageAttention | 补充完整中文 note（9,017c），含背景、方法（INT8 量化、K 矩阵平滑、FP16 累加器）、实验结果（2.1× FlashAttention2 加速）、优点与局限 |
| 2026-06-05 | 2024 | FlashAttention-3 | 补充完整中文 note（14,064c），含背景、方法（内核优化、并行化策略）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SharedAttention | 补充完整中文 note（11,636c），含背景、方法（超越 KV 缓存、共享注意力）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SampleAttention | 补充完整中文 note（10,614c），含背景、方法（采样注意力）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | Quest | 补充完整中文 note（11,428c），含背景、方法（query-aware KV cache sparsity、min/max Key + current Query）、实验结果（7.03× self-attention 加速、2.23× 端到端加速）、优点与局限 |
| 2026-06-05 | 2024 | SparseInfer | 补充完整中文 note（11,931c），含背景、方法（XOR 稀疏性预测、CUDA 内核实现）、实验结果（21% 加速、<1% 精度损失）、优点与局限 |
| 2026-06-05 | 2024 | FlashAttention-2 | 补充完整中文 note（12,853c），含背景、方法（内核优化、并行化策略）、实验结果、优点与局限 |
| 2026-06-05 | 2023 | PagedAttention | 补充完整中文 note（12,556c），含背景、方法（PagedAttention 算法、KV Cache Manager、Copy-on-Write、调度、分布式执行、内核优化）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | Eagle | 补充完整中文 note（11,441c），含背景、方法（特征级自回归、推测采样框架）、实验结果（2.1×-3.8× 加速）、优点与局限 |
| 2026-06-05 | 2024 | TOVA | 补充完整中文 note（10,677c），含背景、方法（MSRNN、Transformer 作为无界 MSRNN、TOVA 压缩策略）、实验结果（4.8× 吞吐提升）、优点与局限 |
| 2026-06-05 | 2024 | ReMoE | 补充完整中文 note（12,102c），含背景、方法（ReLU 路由、自适应 L1 正则化、负载均衡、三阶段训练）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | AVSS | 补充完整中文 note（11,253c），含背景、方法（激活方差、激活稀疏度、AVSS 计算、剪枝策略）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | DeepSeek-V3 | 补充完整中文 note（15,213c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2023 | Dist-Einsum | 补充完整中文 note（9,172c），含背景、方法（通信原语分解、XLA 平台）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | Centauri | 补充完整中文 note（13,565c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2024 | DistGEMM | 补充完整中文 note（11,029c），含背景、方法（分布式 GEMM、Blackwell 架构）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | DHIB73MC | 补充完整中文 note（11,001c），含背景、方法（数据级、模型级、系统级）、实验结果、优点与局限 |
| 2026-06-05 | 2024 | MFA | 补充完整中文 note（9,463c），含背景、方法、实验结果、优点与局限 |
| 2026-06-05 | 2024 | SGLang | 补充完整中文 note（13,440c），含背景、方法（RadixAttention、压缩 FSM、API 推测执行）、实验结果（6.4× 吞吐量提升）、优点与局限 |
| 2026-06-05 | 2026 | Vortex | 新增论文：完整 prototxt 元数据 + 中文 note（12,081c），含背景、方法（vFlow 编程模型、vTensor 抽象、两阶段分解、组合性、AI 代理生成）、实验（AI 代理生成 3.46× 吞吐提升、SGLang 对比 3.60× 吞吐提升、GLM-4.7-Flash 4.7× 加速、MiniMax-M2.7 1.37× 加速、radix top-k 优化 1.49×）、优点与局限 |
| 2026-06-04 | 2026 | ZipServ | 补充完整中文 note（14,955c），含背景、方法（TCA-TBE 编码、ZipGEMM 内核、阶段感知策略、实现细节）、实验（内核级、端到端、内存节省、开销分析）、优点与局限 |
