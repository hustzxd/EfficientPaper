# MiniMax-01: Scaling Foundation Models with Lightning Attention

> MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu

![111](../../blank.jpg)

> **⚠️ 本文档由 AI Agent 自动生成，仅供学术参考，内容基于论文原文。生成时间：2025年。**

---

## 一句话总结

MiniMax-01 是首个将 Lightning Attention（线性注意力）大规模应用于商用级基础模型的系列，通过混合架构（7个线性注意力 + 1个 softmax 注意力）结合 MoE（32 个专家，456B 总参数，45.9B 激活参数），实现百万级上下文窗口，性能媲美 GPT-4o 和 Claude-3.5-Sonnet，同时拥有 20-32 倍更长的上下文处理能力。

---

## 摘要翻译

我们推出 MiniMax-01 系列，包括 MiniMax-Text-01 和 MiniMax-VL-01，它们在处理更长上下文方面具有卓越能力，同时性能与顶级模型相当。核心在于 Lightning Attention 及其高效扩展。为最大化计算能力，我们将其与混合专家（MoE）相结合，构建了一个拥有 32 个专家、4560 亿总参数的模型，其中每个 token 激活 459 亿参数。我们开发了优化的并行策略和高效的计算-通信重叠技术，用于 MoE 和 Lightning Attention。该方法使我们能够在拥有数十亿参数的模型上，跨越数百万 token 的上下文窗口进行高效训练和推理。MiniMax-Text-01 的上下文窗口在训练时可达 100 万 token，在推理时以可承受的成本扩展至 400 万 token。我们的视觉语言模型 MiniMax-VL-01 通过 5120 亿视觉语言 token 的持续训练构建。在标准和内部基准上的实验表明，我们的模型在性能上与 GPT-4o 和 Claude-3.5-Sonnet 等最先进模型相当，同时提供 20-32 倍更长的上下文窗口。我们已在 https://github.com/MiniMax-AI 公开发布 MiniMax-01。

---

## 研究动机

### 问题背景
1. **长上下文需求与现实差距**：当前大多数 LLM 的上下文窗口为 32K-256K token，远不能满足实际需求——无论是处理整本专业书籍、辅助整个编程项目，还是通过多示例充分利用上下文学习。
2. **Transformer 架构的计算瓶颈**：标准 softmax 注意力的 O(n²) 复杂度使得扩展上下文窗口变得极为困难，计算需求增长远超硬件能力。
3. **现有替代方案的局限**：稀疏注意力、线性注意力、状态空间模型（Mamba 系列）、线性 RNN 等方法虽然有理论优势，但在商业规模模型中尚未成功部署。
4. **核心目标**：构建一个在标准基准上性能与顶级商业模型相当，但上下文窗口长一个数量级的模型。

### 创新驱动力
- 需要**在实践中高效**的线性注意力，而非仅理论可行
- 需要在**性能、效率、长上下文**三者间取得平衡
- 需要**从头设计**训练和推理框架，因为现有框架主要为 softmax 注意力优化

---

## 方法（技术细节）

### 1. 模型架构设计

#### 1.1 混合注意力架构
MiniMax-Text-01 采用混合架构，结合 Lightning Attention（线性注意力）和 Softmax Attention：
- **架构模式**：每 7 个 TransNormer 块（线性注意力）后接 1 个 Transformer 块（softmax 注意力），共 80 层
- **注意力头**：64 个头，每个头维度 128
- **GQA（分组查询注意力）**：softmax 注意力层使用 group size 为 8 的 GQA
- **RoPE（旋转位置编码）**：仅应用于 softmax 注意力一半的维度，base frequency 为 10,000
- **隐藏维度**：6144

#### 1.2 Mixture of Experts (MoE)
- **专家数量**：32 个专家
- **路由策略**：Top-2 路由
- **参数规模**：总参数 456B，每个 token 激活 45.9B 参数
- **FFN 隐藏维度**：9216
- **训练策略**：Token-drop 策略（非 dropless），每个专家设有容量上限
- **负载均衡**：Global Router 策略（基于 GShard 改进），通过额外的 allgather 通信同步各 EP 组的 token 数量
- **辅助损失**：$L_{aux} = \alpha_{aux} \cdot \frac{1}{E} \sum_{i=1}^{E} f_i \cdot m_i$

#### 1.3 Lightning Attention（核心创新）
**基本原理**：
- 线性注意力利用"右乘核技巧"将二次复杂度转化为线性复杂度
- 标准 softmax 注意力：$O = \text{Softmax}(QK^T/\sqrt{d})V$，复杂度 O(nd²)
- 线性注意力：$O = \text{Norm}(Q(K^TV))$，复杂度 O(nd²)

**Lightning Attention 的关键技术**：
- **I/O 感知的分块（Tiling）技术**：避免 cumsum 操作
- **块内（intra-block）计算**：使用左乘（left product）
- **块间（inter-block）计算**：使用右乘（right product）
- **递归更新 KV**：$kv_t = kv_{t-1} + k_t v_t^T$
- **最终复杂度**：$O(nd² + nBd)$，其中 B 为块大小
- **推理复杂度**：$O(d²)$（与序列长度无关），而 softmax 注意力为 O(nd²)

**为何选择混合而非纯线性注意力**：
- 纯线性注意力在检索任务（NIAH）上表现不佳，不适合 LLM
- 混合模型（7:1 比例）不仅匹配，而且**超越** softmax 注意力在检索和外推任务上的表现
- 原因分析：softmax 注意力的循环容量为 O(d)，而 lightning attention 的容量为 O(d²/h)（d > h），因此具有更大容量

### 2. 分布式训练优化

#### 2.1 MoE 优化
- **EP-ETP 重叠策略**：引入 Expert Tensor Parallel (ETP) 和 Expert Data Parallel (EDP)
- **通信优化**：将 all-to-all 通信与专家计算重叠，减少 50% 纯通信开销
- **并行策略解耦**：MoE 组件与非 MoE 组件的并行策略完全解耦

#### 2.2 长上下文优化
- **Varlen Ring Attention**：针对数据打包格式优化的 Ring Attention，避免 padding 浪费
- **LASP+（改进的线性注意力序列并行）**：
  - 原始 LASP 存在串行依赖，计算效率低
  - LASP+ 通过本地前缀和 + AllGather + 全局前缀和，将串行计算转为并行
  - 支持 varlen 特性处理不同长度的 batch 输入

#### 2.3 推理优化
- **Batched Kernel Fusion**：融合多个 memory-bound 内核，减少中间结果存储和内存访问
- **分离 Prefill 和 Decoding 执行**：使用两个不同内核和 CUDA 流并行调度
- **多级 Padding**：动态选择计算规模（32/64/128/256），最小化 padding 开销
- **StridedBatchedMatmul 扩展**：利用 NVIDIA cuBLAS 库优化 GEMM 操作
- **最终效果**：MFU > 75%（H20 GPU），在 1M token 序列长度下，softmax 注意力占延迟 95%，lightning attention 不到 12%

### 3. 预训练

#### 3.1 数据
- **语料来源**：学术文献、书籍、网页内容、编程代码
- **质量增强**：基于奖励模型（前代 5B 激活/60B 总参数 MoE 模型）的多维质量评估，聚焦知识深度、实用性、类别分布
- **格式优化**：嵌套文档格式，平衡自然理解与结构一致性
- **数据混合**：平衡采样策略，高权重于高质量内容，同时保持类别多样性
- **Tokenizer**：Byte-level BPE，词表大小 200K，多语言上采样

#### 3.2 训练策略
- **初始化**：Xavier 初始化，DeepNorm 缩放因子 α=(2N)^0.25, β=(8N)^(-0.25)
- **优化器**：AdamW（β1=0.9, β2=0.95, weight decay=0.1）
- **序列长度**：8192
- **批大小**：逐步从 16M → 32M（69B tokens）→ 64M（790B tokens）→ 128M（4.7T tokens），基于临界批大小与训练损失的幂律关系
- **学习率**：线性预热 500 步至 2×10⁻⁴，恒定学习率训练 7.2T tokens，后续调整为 1.3×10⁻⁴（3.2T tokens），快速衰减阶段指数衰减至 3×10⁻⁵
- **总训练量**：超过 10T tokens
- **MoE 辅助损失系数**：0.01

#### 3.3 长上下文扩展（三阶段）
1. **128K**：RoPE 频率 5M，300B tokens，短/中/长数据比例 30%/70%/0%
2. **512K**：RoPE 频率 10M，32B tokens，比例 35%/35%/30%
3. **1M**：RoPE 频率 10M，26B tokens，比例 30%/30%/40%

#### 3.4 模型规格确定
- 目标约束：单机 8×80G GPU，8-bit 量化下处理 1M+ tokens
- 优化问题：在总参数 < 500B、计算预算约束下，最小化损失
- 通过缩放定律拟合和小模型实验，最终确定 45.9B 激活参数、456B 总参数

### 4. 后训练（Post-training）

#### 4.1 SFT（监督微调）
- 多阶段构建：使用领域专家模型，通过迭代 SFT 和 RL 生成高质量响应
- 拒绝采样：多温度采样，通过奖励层次选择最佳示范

#### 4.2 离线 RL（DPO）
- 在 SFT 训练的 prompt 分布上进行
- 生成不同温度的响应，通过奖励模型选择最佳/最差响应构建偏好对

#### 4.3 在线 RL（改进的 GRPO）
- **重要性采样权重裁剪**：解决梯度不稳定性
- **KL 散度优化**：通过方差-偏差权衡分析，降低梯度方差
- **平衡优势估计**：确保正负样本的奖励贡献均衡

#### 4.4 安全对齐
- 安全类别特化 prompt + 真实用户数据收集 + prompt 增强
- 无害奖励模型：基于安全规则，同时融入 helpfulness 原则，防止不合理拒绝

#### 4.5 多阶段长上下文适配训练
1. **Stage I**：短上下文 SFT（8192 tokens），2 epochs
2. **Stage II**：扩展上下文训练（1,032,192 tokens），50% 长上下文 prompt，2 epochs
3. **Stage III**：短上下文 DPO（8192 tokens），1 epoch
4. **Stage IV**：长上下文 DPO（1,032,192 tokens），1 epoch
5. **Stage V**：短上下文在线 RL（8192 tokens），1 epoch

### 5. 视觉语言模型（MiniMax-VL-01）

#### 5.1 架构
- **ViT-MLP-LLM 范式**：ViT 编码器（303M 参数）+ 两层 MLP 投影器（随机初始化）+ MiniMax-Text-01
- **视觉编码器**：ViT-L/14，对比学习（CoCa 方式），336×336 分辨率下 ImageNet-1K 零样本分类准确率 80.55%
- **动态分辨率策略**：输入图像按预定义网格配置列表调整（336×336 至 2016×2016），保留标准缩略图（336×336）
- **特点**：利用长上下文处理能力，直接使用原始高维特征（无降采样），避免信息损失

#### 5.2 训练流程
- **Stage I（模态对齐）**：80B tokens，仅更新图像适配器和视觉编码器，336×336 分辨率
- **Stage II（视觉理解增强）**：420B 多模态 tokens，全部参数更新，多模态数据与文本后训练数据比例 20:1
- **Stage III（用户体验增强）**：44.8B 多模态 tokens，1 epoch，真实用户交互风格数据
- **Stage IV（偏好增强）**：DPO，40,000 图文对，早期停止防止过拟合

#### 5.3 数据规模
- **Caption 数据**：6.94 亿唯一图文对，1.8 亿精炼 caption
- **Description 数据**：1 亿图像，每图约 300 tokens 描述
- **Instruction 数据**：覆盖文档/文本处理（36.1%）、图像分析（20.6%）、物体识别（18.8%）、数学（5.1%）等 13 个类别
- **总训练 tokens**：5120 亿视觉语言 tokens

---

## 实验结果

### 1. 核心文本基准性能（MiniMax-Text-01）

| 任务 | GPT-4o | Claude-3.5-Sonnet | DeepSeek-V3 | MiniMax-Text-01 |
|------|--------|-------------------|-------------|-----------------|
| MMLU* | 85.7 | 88.3 | 88.5 | **88.5** |
| MMLU-Pro* | 74.4 | 78.0 | 75.9 | **75.7** |
| C-SimpleQA | 64.6 | 56.8 | 64.8 | **67.4** |
| IFEval (avg) | 84.1 | 90.1 | 87.3 | **89.1** |
| GPQA* (diamond) | 46.0 | 65.0 | 59.1 | 54.4 |
| MATH* | 76.6 | 74.1 | 84.6 | **77.4** |
| HumanEval | 90.2 | 93.7 | 92.1 | 86.9 |
| Arena-Hard | 92.4 | 87.6 | 91.4 | 89.1 |

**关键发现**：
- C-SimpleQA 上超越所有模型（67.4），中文知识边界更广
- MMLU、IFEval、Arena-Hard 均进入前三
- MATH 优于 GPT-4o、Claude-3.5-Sonnet、Llama-3.1-405B
- GPQA Diamond 超过大多数开源指令微调 LLM 和 GPT-4o

### 2. 长上下文性能

#### RULER 基准（长上下文理解，13 任务平均准确率）
| 模型 | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|------|-----|-----|------|------|------|-------|-------|-------|-----|
| GPT-4o | 0.970 | 0.921 | 0.890 | 0.888 | 0.884 | - | - | - | - |
| Claude-3.5-Sonnet | 0.965 | 0.960 | 0.957 | 0.950 | 0.952 | 0.938 | - | - | - |
| Gemini-1.5-Pro | 0.962 | 0.960 | 0.960 | 0.958 | 0.938 | 0.917 | 0.916 | 0.861 | 0.850 |
| MiniMax-Text-01 | 0.963 | 0.961 | 0.953 | 0.954 | 0.943 | **0.947** | **0.945** | **0.928** | **0.910** |

**关键发现**：
- 128K 以上，MiniMax-Text-01 建立显著优势
- 1M token 上下文下，MiniMax-Text-01（0.910）远超 Gemini-1.5-Pro（0.850）
- 在超过 200K 的长上下文场景中表现显著优于竞争模型

#### LongBench-V2（w/ CoT）
| 模型 | overall | easy | hard | short | medium | long |
|------|---------|------|------|-------|--------|------|
| GPT-4o | 51.4 | 54.2 | 49.7 | 59.6 | 48.6 | 43.5 |
| Claude-3.5-Sonnet | 46.7 | 55.2 | 41.5 | 53.9 | 41.9 | 44.4 |
| MiniMax-Text-01 | **56.5** | **66.1** | **50.5** | **61.7** | **56.7** | 47.2 |

**关键发现**：
- 在 w/ CoT 设置下达到所有评测系统的 SOTA
- 在 w/o CoT 下同样表现优异

#### MR-NIAH（多轮 NIAH）
- MiniMax-Text-01 在英中文评价中均展现强劲性能
- 在大长度输入时性能下降更少，体现长上下文检索的鲁棒性

#### MTOB（长上下文学习，机器翻译）
- eng→kalam (ChrF) 增量：delta half book 45.7，delta full book 45.6，均超越所有竞争模型
- 证明了长上下文训练过程中 in-context learning 能力逐步增强

### 3. 预填充延迟

- MiniMax-Text-01 在 H800 GPU 上的预填充延迟显著低于 GPT-4o、Claude-3.5-Sonnet、Qwen2.5-72B、DeepSeek V3 的 API
- 与 Llama-3-70B（H800）相比，MiniMax-Text-01 在长序列上表现更优

### 4. 视觉语言模型性能（MiniMax-VL-01）

| 任务 | GPT-4o | Claude-3.5-Sonnet | Gemini-2.0-Flash | MiniMax-VL-01 |
|------|--------|-------------------|-----------------|---------------|
| MMMU* | 63.5 | 72.0 | 70.6 | 68.5 |
| ChartQA* | 88.1 | 90.8 | 88.3 | **91.7** |
| DocVQA* | 91.1 | 94.2 | 92.9 | **96.4** |
| OCRBench | 806 | 790 | 846 | **865** |
| AI2D* | 83.1 | 82.0 | 85.1 | 83.3 |
| MathVista* | 62.1 | 65.4 | 73.1 | 68.6 |
| M-LongDoc | 41.4 | 31.4 | 31.4 | 32.5 |

**关键发现**：
- 在 ChartQA（91.7）、DocVQA（96.4）、OCRBench（865）上超越 GPT-4o
- 在长上下文文档理解上表现优异
- 高级数学推理（OlympiadBench）仍有差距
- 在内部用户体验基准上接近 GPT-4o 水平

### 5. 训练速度
- Lightning Attention 在序列长度变化时保持恒定训练速度
- 是唯一一个训练速度超过 FlashAttention2 的线性模型

### 6. 架构消融实验
- **Hybrid-lightning vs. 纯 softmax**：在 MoE 架构中，Hybrid-lightning 在多数基准上优于纯 softmax
- **PostNorm vs. PreNorm**：PostNorm（使用 DeepNorm）在所有评估指标上一致优于 PreNorm
- **MoE vs. Dense**：在相同计算预算下，MoE 模型显著优于 Dense 模型

---

## 优势

1. **超长上下文能力**：训练时 1M tokens，推理时可扩展至 4M tokens，远超 GPT-4o（128K）和 Claude-3.5-Sonnet（200K）
2. **性能与效率的平衡**：456B 总参数但仅 45.9B 激活，在性能上媲美甚至超越顶级商业模型
3. **线性注意力的首次大规模成功部署**：通过混合架构解决了线性注意力的检索能力不足问题
4. **高效推理**：MFU > 75%（H20 GPU），预填充延迟显著低于竞争对手
5. **开源**：模型权重和 API 公开发布，促进社区发展
6. **视觉-语言多模态能力**：MiniMax-VL-01 在多个视觉理解基准上表现优异
7. **完善的训练流程**：从数据处理、预训练、长上下文扩展到后训练对齐，形成了系统化的方法论
8. **创新的分布式训练框架**：EP-ETP 重叠、LASP+、Varlen Ring Attention 等关键技术

---

## 局限

1. **长上下文评估不足**：当前评估数据集主要针对人工或简化场景，实际应用中的长文本推理评估有限
2. **架构残余**：仍保留 1/8 的 vanilla softmax 注意力，限制了完全消除计算开销的可能性
3. **复杂编程任务表现不足**：预训练阶段的编程数据集仍有限，高级编程任务性能有待提升
4. **部分基准表现**：GPQA Diamond（54.4）低于 Claude-3.5-Sonnet（65.0）和 Gemini-2.0-Flash（62.1）
5. **VL 模型高级数学推理**：OlympiadBench（24.2）低于 Gemini-2.0-Flash（46.1）
6. **硬件依赖**：需要大量 GPU 资源（1500-2500 个 H800 GPU），限制了可复现性
7. **纯线性注意力的限制**：虽然混合架构效果良好，但纯线性注意力在 LLM 中仍不适用（检索能力不足）

---

## 与 EfficientPaper 相关的研究方向

### 1. 高效注意力机制
- **线性注意力的规模化部署**：MiniMax-01 证明了线性注意力可以大规模商用，为后续研究提供了重要参考
- **混合注意力架构设计**：7:1 的 lightning/softmax 比例是一个有效的平衡点，但是否可以通过更好的设计消除 softmax 注意力
- **I/O 感知的高效实现**：Lightning Attention 的 tiling 技术和 kernel 优化对硬件效率至关重要

### 2. 高效 MoE 架构
- **负载均衡策略**：Global Router 和 EP-ETP 重叠策略对 MoE 训练效率的提升
- **专家并行优化**：ETP/EDP 并行策略可以推广到其他 MoE 模型

### 3. 高效长上下文处理
- **Varlen Ring Attention**：解决数据打包格式下的 ring attention 问题
- **LASP+**：将串行计算转为并行，大幅提升线性注意力的训练效率
- **推理优化技术**：Kernel 融合、多级 Padding、分离 Prefill/Decoding 等技术

### 4. 高效训练策略
- **三阶段长上下文扩展**：通过渐进式数据混合和 RoPE 频率调整实现上下文窗口扩展
- **数据效率研究**：重复感知的实验框架、平衡采样策略
- **临界批大小**：基于训练损失的幂律关系动态调整批大小

### 5. 多模态高效架构
- **ViT-MLP-LLM 范式**：轻量视觉编码器 + MLP 投影器 + 长上下文 LLM
- **动态分辨率策略**：利用长上下文处理能力避免降采样信息损失

### 6. 高效对齐方法
- **多阶段对齐训练**：SFT → DPO → 在线 RL 的完整流程
- **改进的 GRPO**：重要性采样权重裁剪、KL 散度优化、平衡优势估计

### 7. 未来研究方向
- 完全消除 softmax 注意力的架构
- 更高效的长上下文评估方法
- 更强的编程和推理能力
- 更低资源消耗的训练和推理方案

---

## 参考信息

- **论文标题**：MiniMax-01: Scaling Foundation Models with Lightning Attention
- **发表时间**：2025年1月14日
- **arXiv**：https://arxiv.org/abs/2501.08313v1
- **代码**：https://github.com/MiniMax-AI/MiniMax-01
- **机构**：MiniMax
- **关键词**：structure_design, Lightning Attention, MoE, Linear Attention, Long Context
