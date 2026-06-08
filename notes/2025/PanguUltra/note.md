# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs

> Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, Zhicheng Liu

![](../../blank.jpg)

---

> **⚠️ 生成声明：本 note 由 AI Agent 自动生成，基于 arXiv 论文全文提取与分析。生成时间：2025年。内容仅供参考，如有疏漏请以原文为准。**

---

## 一句话总结

Pangu Ultra 是华为盘古团队在昇腾 NPU 上训练的 135B 参数稠密 Transformer 语言模型，通过深度缩放三明治归一化（Depth-Scaled Sandwich-Norm）和微初始化（TinyInit）实现稳定训练，在 13.2 万亿 token 上预训练后，在多项基准上超越 Llama 405B 等稠密模型，并与 DeepSeek-R1 等稀疏大模型达到竞争水平。

---

## 摘要翻译

我们提出了 Pangu Ultra，一个拥有 1350 亿参数、基于稠密 Transformer 模块、在昇腾神经处理单元（NPU）上训练的大语言模型（LLM）。尽管近年来 LLM 领域在规模和能力方面取得了前所未有的进展，但训练如此大规模的模型仍然面临显著的优化和系统挑战。为了稳定训练过程，我们提出了**深度缩放三明治归一化**，有效消除了深层模型训练过程中的 loss spike。我们在 13.2 万亿多样且高质量的 token 上预训练模型，并在后训练阶段进一步增强其推理能力。为了高效执行如此大规模的训练，我们利用 8192 块昇腾 NPU 并结合一系列系统优化。在多个多样基准上的评估表明，Pangu Ultra 显著推进了稠密 LLM（如 Llama 405B 和 Mistral Large 2）的能力前沿，甚至与拥有更多参数的稀疏模型 DeepSeek-R1 取得了竞争性的结果。我们的探索证明昇腾 NPU 能够高效、有效地训练超过 1000 亿参数的稠密模型。我们的模型和系统将向商业客户开放。

---

## 研究动机

1. **稠密 vs 稀疏模型之争**：当前大模型训练中，MoE 稀疏模型（如 DeepSeek）因参数效率高而成为主流选择，但稠密模型在推理效率和部署简便性方面具有优势。本文旨在探索稠密模型在大规模参数下的性能上限。
2. **深度与参数的双重挑战**：模型深度对推理能力有显著影响，但增加深度会带来训练不稳定性（loss spike）。同时，训练 135B 稠密模型需要协调数千个 AI 处理器，面临系统效率挑战。
3. **国产硬件验证**：华为希望证明昇腾 NPU 能够支撑百亿级参数稠密模型的高效训练，推动国产 AI 芯片生态。

---

## 方法（技术细节）

### 2.1 模型架构

- **基础结构**：类似 Llama 3 的稠密 Transformer
- **参数规模**：135B 参数，94 层
- **隐藏维度**：12,288
- **FFN 中间维度**：28,672（SwiGLU 激活）
- **注意力机制**：Group Query Attention (GQA)，96 个 query head，8 个 KV head，有效减少 KV-cache 大小

### 2.2 深度缩放三明治归一化（Depth-Scaled Sandwich-Norm, DSSN）

这是本文的核心技术创新之一，解决了深层稠密模型训练不稳定的问题：

- **三明治归一化**：在每个子层（Attention/FFN）的输出上应用 LayerNorm（在残差连接之前），即 sandwich-norm。相比 Pre-LN，它同时对子层输出进行前后归一化。
- **深度缩放**：将 LayerNorm 的可训练 gamma 参数初始化为与网络深度的平方根成反比的值：

  ```
  h ← h + Norm(γ_attn, ATTN(Norm(h)))
  γ_attn = c_attn / √L
  h ← h + Norm(γ_mlp, MLP(Norm(h)))
  γ_mlp = c_mlp / √L
  ```

  其中 L 为层数，c_attn 和 c_mlp 分别设为 0.283 和 0.432。

- **效果**：有效控制梯度波动，消除 loss spike，加速收敛。消融实验表明，DSSN 相比 Pre-LN 在 EN basic、ZH basic、LAMBADA、WPLC 上均有提升。

### 2.3 微初始化（TinyInit）

- 标准的 Transformer 初始化使用 N(0, 2/(5d)) 或 N(0, 1/(5dL))。
- **TinyInit** 提出同时按深度和宽度缩放初始化标准差：N(0, 1/(2dL))
- 假设：更一致的参数尺度有助于优化和收敛。
- 嵌入层标准差设为 0.5（而非接近 1），实验证实效果良好。
- 消融实验：在 102B token 训练后，TinyInit 在多个基准上优于传统初始化（如 C-Eval 从 0.476 提升至 0.524，MMLU 从 0.473 提升至 0.502）。

### 2.4 分词器

- 采用**领域感知词汇表**策略：分别对中文、英文、代码、数学等领域进行独立频率分析，生成领域特定词表后合并去重。
- 词表大小：153,376 个唯一 token。
- 分布：英文 44.35%，中文 26.77%，其他 19.93%，拉丁语 2.94%，阿拉伯语 1.80%，韩语 1.78%，数学 1.39%，日语 1.04%。

### 2.5 预训练流程

- **数据规模**：13.2 万亿 token（13.2T）
- **三阶段数据配方**：
  - **通用阶段（12T tokens）**：分两个子阶段，7.4T + 4.6T，以英文和中文网页、书籍、百科等为主，第二子阶段使用更高质量数据。
  - **推理阶段（0.8T tokens）**：数学和代码数据占比超过 60%，引入 LLM 生成的合成数据。
  - **退火阶段（0.4T tokens）**：指令数据占约 20%，构建短链和长链 CoT 响应，巩固知识和推理技能。
- **训练策略**：AdamW 优化器（β1=0.9, β2=0.95, weight decay=0.1），梯度裁剪 1.0。
  - 0T–7.4T：序列长度 4K（RoPE base=1e4），batch size 从 1024 增至 2048，学习率余弦衰减 1e-4→1e-5。
  - 7.4T–12.0T：序列长度 4K，batch size 2048，学习率固定 1e-5。
  - 12.0T–12.8T：序列长度 8K（RoPE base=1e5），batch size 1536，学习率余弦衰减 1e-5→7.5e-6。
- **数据质量**：结合规则和模型评估，高质量数据的采样概率更高。消融实验表明，低质量数据需要 1.6× 的 token 才能达到同等性能。

### 2.6 长上下文扩展

- **目标**：支持最大 128K 上下文长度。
- **方法**：增加 RoPE 基础频率（而非 YaRN）。
  - 8K→32K：RoPE base=1.6e6，batch size 384，学习率 7.5e-6。
  - 32K→128K：RoPE base=2.56e7，batch size 96，学习率 7.5e-6。
- 通过 "Needle In A Haystack"（NIAH）离线评估选择最优基础频率。

### 2.7 后训练

- **SFT 冷启动**：使用精心策划的指令数据，涵盖通用问答、AIGC、代码、数学、逻辑推理等。
- **RL 强化学习**：基于结果奖励信号，采用混合奖励系统（确定性奖励 + 模型评估），针对数学、代码、通用问题求解。
- 后训练数据中约 6/7 为推理任务（数学、代码、逻辑）。
- 实现了针对昇腾基础设施优化的延迟容忍 RL 框架。

### 2.8 训练系统

- **硬件**：8192 块昇腾 NPU，每节点 8 块，64GB 内存，HCCS 全互联（节点内），200Gbps RoCE（节点间）。
- **并行策略**：
  - 128-way DP + ZERO（减少参数和优化器状态的内存开销）
  - 8-way TP（利用节点内高带宽）
  - 8-way PP（利用节点间连接）
  - 6-way VPP（虚拟流水线阶段）：将 PP 气泡比率从 30.45% 降至 6.8%。
- **系统优化**：
  - **MC2**（Merged Compute and Communication）：融合 MatMul 与通信操作，减少通信延迟。
  - **NPU Fusion Attention (NFA)**：针对昇腾 NPU 的自注意力融合算子，支持 reset attention mask 策略，利用 2048×2048 因果掩码模板。
  - **子序列分区（Context Parallelism）**：提出负载均衡的分区策略，每个 rank 计算两个子序列块。
  - **快速掩码生成与数据复用**：用 NPU 算子计算 attention mask（而非 CPU），跨 VPP 阶段共享掩码。
  - 其他融合：RMSNorm、SwiGLU、RoPE、梯度累积、PP 通信等。
- **训练效率**：MFU 超过 52%（基于 8192 NPU）。

---

## 实验结果

### 预训练基准（Base Model）

| 基准 | Qwen2.5-72B | Llama-3.1-405B | DeepSeek-V3 | Pangu Ultra |
|------|-------------|----------------|-------------|-------------|
| MMLU (5-shot) | 85.0 | 84.4 | 87.1 | **85.4** |
| MMLU-Pro (5-shot) | 58.3 | 52.8 | 64.4 | **63.1** |
| GSM8K (8-shot) | 88.3 | 83.5 | 89.3 | **89.3** |
| MATH (4-shot) | 54.4 | 49.0 | 61.6 | **62.5** |
| C-Eval (5-shot) | 89.2 | 72.5 | 90.1 | **90.3** |
| CMMLU (5-shot) | 89.5 | 73.7 | 88.8 | **91.7** |
| HumanEval (0-shot) | 53.0 | 54.9 | 65.2 | **81.1** |
| HellaSwag (10-shot) | 84.8 | 89.2 | 88.9 | **99.0** |
| ARC-Challenge (25-shot) | 94.5 | 95.3 | 95.3 | **97.0** |

**关键发现**：
- Pangu Ultra 在大多数英文通用基准上达到 SOTA，所有中文基准上超越 Qwen2.5-72B 和 DeepSeek-V3。
- 在代码任务（HumanEval）上表现突出（81.1）。
- 仅使用 Llama 405B 约 29% 的训练 FLOPs，即在大多数基准上超越 Llama 405B。
- 在数学推理上与 DeepSeek-V3 竞争力强（MATH 62.5 vs 61.6）。

### 后训练与推理能力

| 模型 | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench | ArenaHard | MMLU-pro |
|------|-----------|----------|--------------|---------------|-----------|----------|
| GPT-4o-0513 | 9.3 | 74.6 | 49.9 | 32.9 | 80.4 | 72.6 |
| Qwen2.5-72B | 16.0 | 83.1 | 49 | 27.6 | 81.2 | 72.0 |
| DeepSeek-R1 | 79.8 | 97.3 | 71.5 | 65.9 | 92.3 | 84.0 |
| Hunyuan-T1 | 79.8 | 96.2 | 69.3 | 64.9 | 91.9 | 87.2 |
| **Pangu Ultra** | **80.8** | **97.4** | **74.2** | **66.5** | 91.5 | **84.4** |

**关键发现**：
- Pangu Ultra 在推理基准上达到 SOTA：AIME 2024（80.8）、MATH-500（97.4）、GPQA Diamond（74.2）、LiveCodeBench（66.5）。
- 与 DeepSeek-R1 竞争性极强，在多个推理任务上甚至超越。
- 在一般语言理解（ArenaHard、MMLU-pro）上保持强竞争力。

### 消融实验

1. **DSSN vs Pre-LN**（13B 模型，300B tokens）：DSSN 在所有基准上优于 Pre-LN，消除了 loss spike，梯度范数更稳定。
2. **DSSN vs 普通 Sandwich-Norm**（1.6B 模型，94 层）：普通 sandwich-norm 仍有 loss spike，DSSN 训练平稳且损失更低。
3. **TinyInit vs 传统初始化**（102B tokens）：TinyInit 在 EN basic、ZH basic、LAMBADA、WPLC、C-Eval、MMLU、BIG-bench 上均提升。
4. **训练过程无 loss spike**：整个 13.2T 预训练过程中未出现 loss spike。

---

## 优势

1. **稠密模型性能突破**：135B 稠密模型在多项基准上超越 405B 稠密模型和 671B 稀疏模型（DeepSeek-V3），证明了稠密模型的潜力。
2. **训练稳定性**：DSSN + TinyInit 组合彻底消除了 loss spike，这在大模型训练中极为罕见且关键。
3. **高效训练**：MFU 超过 52%，通过虚拟流水线（VPP）、MC2、NFA 等系统优化大幅提高硬件利用率。
4. **国产硬件可行性**：证明昇腾 NPU 能够训练百亿级参数稠密模型，为国产 AI 芯片提供了实际验证。
5. **中国基准优势**：在中文任务上表现尤为突出，超越 Qwen2.5 和 DeepSeek-V3。
6. **训练效率高**：仅用 Llama 405B 约 29% 的训练 FLOPs 即可达到类似甚至更好的性能。
7. **三阶段预训练策略**：从通用知识→推理能力→知识巩固的渐进式训练策略设计合理。
8. **分词器创新**：领域感知词汇表确保了数学、代码等专业领域的覆盖。

---

## 局限

1. **代码公开性不足**：未开源代码和模型权重（仅向商业客户开放），限制了学术社区的复现和研究。
2. **与 DeepSeek-R1 差距**：虽在部分推理任务上超越 DeepSeek-R1，但在通用语言理解（如 BBH、ArenaHard）上仍有差距。
3. **规模限制**：135B 参数仍小于 Llama 405B（参数量）和 DeepSeek-V3（671B 总参数，37B 激活参数），稠密模型的扩展性仍有待验证。
4. **后训练细节不充分**：RL 框架和混合奖励系统的具体实现细节留待后续报告。
5. **长上下文能力评估有限**：虽然支持 128K 上下文，但缺乏系统性的长上下文基准评估（如 RULER、LongBench 等）。
6. **硬件依赖**：训练系统深度绑定华为昇腾 NPU，迁移性受限。
7. **部分基准表现不如预期**：如 DROP（61.0 vs 89.0）、CMath（78.2 vs 90.7）等任务上表现不如 DeepSeek-V3。
8. **多语言覆盖有限**：虽然支持多语言，但日语、阿拉伯语等低资源语言的覆盖可能不足。

---

## 与 EfficientPaper 相关的研究方向

1. **模型结构设计（structure_design）**：DSSN 和 TinyInit 为深层稠密模型的稳定训练提供了新范式，可与 EfficientPaper 中其他结构设计方法（如高效注意力、稀疏化）对比。
2. **训练稳定性**：Loss spike 的消除是大模型训练中的关键问题，DSSN 的深度缩放归一化方案可推广到其他深层模型。
3. **并行训练优化**：虚拟流水线（VPP）、MC2 等系统优化方法对高效分布式训练有重要参考价值。
4. **稠密 vs 稀疏模型**：本文证明了稠密模型在大规模参数下仍具竞争力，与 MoE 稀疏模型形成对比，可纳入 EfficientPaper 的方法对比框架。
5. **数据质量与课程学习**：三阶段预训练策略和数据质量评估方法对数据高效利用有参考价值。
6. **长上下文训练**：子序列分区和快速掩码生成等方法对长序列高效训练有实际意义。
7. **国产 AI 芯片训练**：昇腾 NPU 的大规模训练经验为国产芯片在 LLM 训练中的应用提供了数据支撑。
8. **与基准方法的对比**：本论文 baseline 为 DeepSeek-R1，可在 EfficientPaper 中追踪其与 Qwen、Llama、Gemma 等系列的性能演进。

---

*注：本 note 由 AI Agent 自动生成，基于 arXiv 论文（2504.07866v2）全文提取与分析。所有中文翻译和分析仅供参考，如有疏漏请以原文为准。*
