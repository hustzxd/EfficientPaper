# OmniDrop: Layer-wise Token Pruning for Omni-modal LLMs via Query-Guidance

> Yeo Jeong Park, Hyemi Jang, Minseo Choi, Jongsun Lee, Jooyoung Choi, Yongkweon Jeon

![111](../../blank.jpg)

## Abstract

Omni-modal large language models have demonstrated remarkable potential in holistic multimodal understanding; however, the token explosion caused by high-resolution audio and video inputs remains a critical bottleneck for real-time applications and long-form reasoning. Existing omni-modal token compression methods typically prune tokens at the input embedding level, relying on audio-video similarity or temporal co-occurrence as proxies for semantic relevance. In practice, such assumptions are often unreliable. To address this limitation, we propose OmniDrop, a training-free, layer-wise token pruning framework that progressively prunes audiovisual tokens within the LLM decoder layers rather than at the input-level, allowing early layers to preserve sufficient omni-modal information fusion before aggressively removing tokens in deeper layers. We further utilize text queries as guidance for modality-agnostic and task-adaptive token pruning. We also introduce a temporal diversity score that encourages balanced token survival to preserve global temporal context. Experimental results across various audiovisual benchmarks demonstrate that OmniDrop outperforms all baselines by up to 3.58 points while reducing prefill latency by up to 40% and memory usage by up to 14.7%.


---

*以下总结由 MiMo 生成：*

这篇论文针对全模态大语言模型中因高分辨率音视频输入导致的令牌爆炸问题，提出了一种名为OmniDrop的训练无关分层令牌剪枝框架。该方法通过在LLM解码器层内逐步剪枝音视频令牌，并利用文本查询作为跨模态任务自适应的剪枝指导，同时引入时间多样性分数以保留全局时间上下文。实验表明，OmniDrop在各类音视频基准测试中性能优于基线模型最高3.58分，同时预填充延迟降低40%，内存使用减少14.7%。

---

## 论文详细总结

> 由 GPT 自动生成，请人工核验。

### 1. 研究背景与动机

Omni-modal LLM 同时处理文本、音频和视频，适合实时多模态助手、视频理解和长程音视频推理。但高分辨率音视频会造成严重 **token explosion**：论文提到 Qwen2.5-Omni 中 1 分钟视频会产生超过 10k 个音视频 token，导致 prefill 计算和显存开销快速增长。

已有 Omni-modal token compression 方法多在 **input embedding level** 做剪枝，通常假设：

- 音频和视频 token 在共享 embedding 空间中相似就代表语义相关；
- 同一时间段出现的音频和视频具有相同语义上下文；
- 音频重要性可以作为视频 token 选择的 proxy。

论文通过 PCA、cosine similarity 分布和 OmniZip ablation 说明这些假设不可靠：音频/视频 token 在进入 LLM 前仍分布在不同子空间，跨模态 embedding similarity 甚至不优于 random selection。因此，作者主张不要在输入层过早剪枝，而应等 LLM decoder 中逐步完成 omni-modal fusion 后，再做 layer-wise token pruning。

### 2. OmniDrop 核心思想

OmniDrop 是一个 **training-free、layer-wise、query-guided token pruning** 框架。它不是在输入层一次性删除音视频 token，而是在 LLM decoder 内逐层渐进剪枝：早期层保留更多 token 以完成音视频融合，中后层根据文本 query 对音频/视频 token 的注意力相关性删除不重要 token。

一句话概括：**用文本 query 作为任务自适应信号，在 decoder 层内渐进剪掉对当前问题不重要的音视频 token，同时用 temporal diversity 保留全局时间上下文。**

### 3. 关键技术

| 技术 | 说明 |
|------|------|
| **Progressive Layer-wise Pruning (PLP)** | 按层逐步增加剪枝强度，早期层保留较多音视频 token，中后层更激进剪枝；比 input-level pruning 更适合保留早期 omni-modal fusion。 |
| **Query-guided token importance** | 使用 text-to-audiovisual attention 估计音频/视频 token 与当前 query 的相关性，实现 modality-agnostic、task-adaptive pruning。 |
| **Temporal Diversity Score (TDS)** | 对距离高相关关键片段更远的 token 给予额外分数，避免只保留局部高 attention token 而丢失全局时间上下文。 |
| **Training-free inference acceleration** | 不训练模型、不改模型结构，只在推理阶段基于 attention 和固定 schedule 选择保留 token。 |
| **Task-adaptive modality retention** | 不需要显式任务标签，query 会自然调节音频/视频保留比例：音频识别更偏音频，场景识别更偏视频，音源定位更均衡。 |

### 4. 实验结果

| 指标 | 结果 |
|------|------|
| 主模型 | Qwen2.5-Omni-7B / 3B |
| 评测任务 | VideoMME、WorldSense、AVUT 等音视频理解 benchmark |
| 对比方法 | Full token baseline、OmniZip、DASH |
| 最低平均保留率 | 可压到 **20%** token retention |
| 性能收益 | 相比 baseline 最高提升 **3.58 points** |
| VideoMME 结果 | 30% retention 下超过 full-token baseline：7B 为 **66.52 vs. 64.67**，3B 为 **63.07 vs. 62.51** |
| AVUT 结果 | 7B、20% retention 下 OmniDrop **63.67**，优于 DASH **60.09**，差距 **3.58 points** |
| 压缩鲁棒性 | 从 30% 降到 20% retention，平均性能下降仅 **0.18** (7B) / **0.47** (3B) points |
| Prefill latency | WorldSense 上降低最多 **39.9%** (7B) / **28.1%** (3B) |
| GPU memory | WorldSense 上降低 **11.3%** (7B) / **14.7%** (3B) |

### 5. Ablation 结论

| 组件 | 结论 |
|------|------|
| **PLP vs. input/intra pruning** | 30% retention 下，intra-pruning 在 WorldSense / AVUT 为 44.33 / 59.34，而 PLP 恢复到约 46.53 / 63.90，说明层内渐进剪枝明显更稳。 |
| **Sigmoid schedule** | 在 20% aggressive pruning 下，sigmoid schedule 优于 exponential schedule，更适合“早期保留、中后期剪枝”。 |
| **TDS** | 30% 和 20% retention 下均提升 WorldSense / AVUT，且压缩越激进收益越明显。 |
| **Text guidance vs. audio guidance** | 用 text query 指导明显优于 audio-to-video guidance；20% retention 下 WorldSense 从 43.25 提升到 46.19，AVUT 从 56.57 提升到 60.55。 |

### 6. 核心贡献

1. 指出 Omni-modal LLM 中 **input-level audio-video similarity 并不是可靠剪枝信号**，音频和视频在 LLM 前未充分对齐。
2. 提出 **training-free layer-wise pruning**，避免在输入层过早丢失跨模态融合所需信息。
3. 用 **text query** 作为任务自适应 relevance signal，同时支持音频/视频 token 的统一选择。
4. 引入 **Temporal Diversity Score**，在高压缩下保留长视频/长音频的全局时间覆盖。
5. 在多种音视频 benchmark 上同时提升 accuracy、prefill latency 和显存效率。

### 7. 局限性与启发

- **依赖文本 query**：若只有音频/视频输入而无文本提示，当前剪枝准则不直接适用。
- **超参数经验化**：pruning schedule 和若干超参数由经验设定，未来可用 calibration data 学习或联合优化。
- **对当前研究的启发**：OmniDrop 与 SPEED/UniPrefill 类似，说明不同 layer/phase/token 的冗余程度不同；可以迁移到 long-context LLM serving 中，研究 query-guided、layer-wise、training-free 的 KV cache eviction/offload 策略。
