# OmniKV

![111](cover.jpg)

> **一句话总结**：OmniKV 利用"层间注意力相似性"这一关键洞察，在不丢弃任何 token 的前提下，仅对少数"过滤层"计算完整注意力来动态选择重要上下文子集，从而实现 1.68 倍解码加速且零性能损失，同时可将 KV cache 内存占用降低 75%，将单张 A100 支持的最大上下文长度从 128K 扩展到 450K。

---

## 摘要翻译

在大语言模型（LLM）的长上下文推理阶段，大量 GPU 显存被分配给 KV cache，且内存占用随序列长度线性增长。为了缓解 KV cache 的 GPU 内存开销，先前研究基于注意力稀疏性丢弃不重要的 token。然而，作者认为注意力分数无法反映 token 在后续生成迭代中的重要性，因为注意力分数仅基于当前隐藏状态计算。因此，本文提出 OmniKV，一种无需丢弃 token 且无需训练的推理方法，实现了 1.68 倍加速且无性能损失。OmniKV 与 offloading 技术高度兼容，可将 KV cache 内存占用降低高达 75%。OmniKV 的核心创新洞察是：在单次生成迭代中，连续层之间识别出的重要 token 具有高度相似性（层间注意力相似性）。大量实验表明，OmniKV 在多个基准测试上实现了最先进的性能，尤其在思维链（Chain-of-Thoughts）场景中具有显著优势。OmniKV 将单张 A100 支持 Llama-3-8B 的最大上下文长度从 128K 扩展到 450K。

---

## 研究动机

### 问题背景
- LLM 长上下文推理中，KV cache 占据大量 GPU 显存。例如 Llama-3-8B 在 batch size=8、上下文长度 128K 时，KV cache 单独占用超过 134GB 显存。
- 现有方法（如 H2O、SnapKV 等）基于注意力分数丢弃"不重要" token 以减少内存占用。

### 核心问题
- **注意力分数的局限性**：注意力分数仅反映 token 与当前推理步骤的相关性，无法预测 token 在后续生成迭代中的重要性。
- **多步推理中的动态性**：在 Chain-of-Thought 等多步推理场景中，不同推理步骤需要的关键 token 完全不同（论文图 1b 示例：两个推理步骤中最重要的 12 个 token 几乎完全不同）。
- **丢弃 token 的风险**：在预填充阶段被丢弃的 token 可能是后续推理步骤所需的关键信息，导致不可逆的信息损失。

### 动机总结
现有 KV cache 压缩方法在多步推理场景中可能因丢弃关键 token 而导致性能下降，因此需要一种不丢弃任何 token 的动态上下文选择方法。

---

## 方法（技术细节）

### 三大核心洞察

#### 1. 层内注意力稀疏性（Intra-Layer Attention Sparsity）
- LLM 的注意力矩阵天然具有稀疏性，仅需关注少量 token 即可生成几乎等价的输出。
- 这一特性为 OmniKV 通过稀疏注意力减少计算量提供了理论基础。

#### 2. 层间注意力相似性（Inter-Layer Attention Similarity）—— 核心创新
- **定义**：在单次生成迭代中，连续层之间识别出的重要 token 集合具有高度相似性。
- **量化**：用某一层选择的重要 token 子集在后续层的累积注意力分数之均值来衡量相似度。
- **关键发现**（图 1a）：即使相隔 16 层，层间注意力相似度仍可达到 0.85-0.95 的高水平。
- **"过滤层"（Filter Layer）**：某些层具有更强的"过滤能力"，即它们选择的重要 token 子集在后续层中的累积注意力分数更高。这些层被选为"过滤层"。
- **任务无关性**（图 6a）：过滤层的过滤能力不依赖于具体任务，而是模型本身的固有特性，因此超参数 L 可以一次确定后适用于所有任务。

#### 3. Token 间注意力变异性（Inter-Token Attention Variability）
- 在多步推理场景中，不同生成迭代需要的关键 token 集合差异巨大（图 1b）。
- 如果仅保留预填充阶段的高注意力 token（如 H2O），后续推理步骤中被丢弃的 token 可能成为关键信息。
- **因此 OmniKV 保留所有 token 的 KV cache，不做任何丢弃**。

### 方法框架

OmniKV 由两个核心模块组成：**Context Bank（上下文存储库）** 和 **Context Selector（上下文选择器）**。

#### Context Bank（上下文存储库）

**Prefill 阶段**：
- 所有层执行完整注意力，生成完整 KV cache：$\{K_i, V_i\}_{i=1}^L$
- 根据层间注意力相似性，将大多数"非过滤层"的 KV cache 卸载到 CPU 内存
- 保留少数"过滤层"（集合 L，大小 m ≤ 3）的完整 KV cache 在 GPU 上

**过滤层选择策略**：
- 设置超参数 L = {层索引集合}，例如 Llama-3-8B 使用 {2, 8, 18}
- 浅层（< L₀）因为稀疏性较低，也执行完整注意力但不做选择
- 过滤层之间的层（Li < l < Li+1）使用前一过滤层选择的 token 子集执行稀疏注意力

**核心公式**：

$$T_i = \begin{cases} \text{ContextSelector}(h_i^w, K_i) & \text{if } i \in L \\ T_{i-1} & \text{otherwise} \end{cases}, \quad i \geq L_0$$

$$\text{out}_i = \begin{cases} \text{Attention}_i(h_i^l, K_i, V_i) & \text{if } i \in L \text{ or } i-1 \in L \text{ or } i < L_0 \\ \text{Attention}_i(h_i^l, K_i[T_i], V_i[T_i]) & \text{otherwise} \end{cases}$$

其中：
- $h_i^w$ 是观察窗口的隐藏状态
- $h_i^l$ 是最后一个 token 的隐藏状态
- $T_i$ 是过滤层 $i$ 选择的重要 token 索引

**Packed Load（打包加载）**：
- 由于过滤层之间的非过滤层共享相同的 token 索引 T，可将连续稀疏注意力层的 KV cache 在最近的前一过滤层一次性从 CPU 加载到 GPU
- 每个解码迭代仅需 ≤ 3 次 GPU-CPU 数据传输，大幅减少 PCIe 传输开销

#### Context Selector（上下文选择器）

在过滤层 L 中，基于注意力分数动态选择 top-k 重要 token。

**计算流程**：
1. 以观察窗口（observation window）的 token 作为 query 状态
2. 以完整上下文作为 key 状态
3. 计算注意力分数 $A_i$
4. 通过 reduce-max 获取跨注意力头的最大分数
5. 使用加权向量 α 进行加权求和得到分数向量 $S_i$
6. 通过 top-k 选择重要 token 集合 $T_i$

$$S_i = \sum_{j=0}^{|h_i^w|-1} \alpha_j \max_{0 \leq h < H} A_i[h, j]$$

**三种选择器**：
1. **Uniform（均匀）**：$\alpha = \{1\}$，窗口中每个 token 贡献相同
2. **Exponential（指数）**：$\alpha = \{2^{i-|h_w|}\}$，靠近窗口末尾的 token 贡献更高
3. **Last Token（末尾 token）**：$\alpha = \{0, ..., 0, 1\}$，仅考虑窗口最后一个 token 的注意力分数

**实验结论**：
- 单步推理：Exponential 和 Last Token 表现最佳
- 多步推理：Last Token 最佳
- Last Token 最简单且延迟最低，与 LLM 预训练范式一致
- 后续实验主要使用 Last Token 选择器

### 关键设计特点

1. **无 token 丢弃（Drop-free）**：保留所有 token 的 KV cache，确保多步推理性能不受影响
2. **无需训练（Training-free）**：完全基于推理阶段的注意力分数动态选择
3. **与 offloading 高度兼容**：可大幅降低 KV cache 内存占用
4. **与现有技术正交**：可与 KV cache 量化（如 KIVI、SmoothQuant）等方法结合使用
5. **支持多种并行策略**：与 Tensor Parallelism、Pipeline Parallelism、Context Parallelism 兼容

---

## 实验结果

### 实验设置

**模型**：
- Llama-3-8B-262K
- Yi-9B-200K
- Llama-3.1-70B-Instruct
- Llama-3.1-405B（附录扩展）

**基准测试**：
- **单步推理**：LongBench（18个子任务）、InfiniteBench（10个子任务）
- **多步推理**：2WikiMQA、HotpotQA、2StageRetr（作者提出的新基准）
- **Needle-in-a-Haystack**：最大 512K 上下文

**基线方法**：
- H2O（注意力分数丢弃）
- InfLLM（基于块的检索）
- StreamingLLM（滑动窗口 + 初始 token 保留）
- Full Attention（完整注意力，作为理论上限）

**硬件**：NVIDIA A100 80GB GPU

### 核心结果

#### 单步推理性能

**LongBench（表 1）**：

| 模型 | 方法 | %Mem | 平均分 |
|------|------|------|--------|
| Llama-3-8B-262K | Full Attention | 100% | 39.2 |
| | H2O | 30% | 36.8 |
| | InfLLM | ~30% | 35.0 |
| | StreamingLLM | 30% | 28.1 |
| | **OmniKV w/ last** | **30%** | **38.5** |
| Yi-9B-200K | Full Attention | 100% | 41.9 |
| | **OmniKV w/ exp** | **30%** | **41.6** |
| Llama-3.1-70B | Full Attention | 100% | 49.2 |
| | **OmniKV w/ exp** | **20%** | **48.7** |

**关键发现**：
- OmniKV 在 30% 内存预算下，性能接近甚至超过原始模型
- 在 Llama-3.1-70B 上，OmniKV 显著优于所有基线（20% 内存预算下：48.7 vs H2O+ 39.9 vs InfLLM 39.5）
- OmniKV 的性能稳定性使其可直接应用于实际场景，无需额外测试

**InfiniteBench（表 2）**：
- OmniKV w/ last 在 Llama-3-8B-262K 上 30% 内存预算下达到 37.4（接近全注意力的 38.1）
- 在 Yi-9B-200K 上达到 38.2（接近全注意力的 38.3）
- 在 Passkey 和 Number Retrieval 任务上保持 100% 准确率

#### 多步推理性能

**2WikiMQA、HotpotQA、2StageRetr（图 3）**：
- OmniKV 在所有预算下均获得最佳性能
- 在 2StageRetr 任务中，H2O 的准确率无法超过其预算比例（因为没有先验知识，只能随机保留 key-value 对）
- OmniKV 基于观察窗口动态选择最相关上下文，因此表现优异

#### 延迟与效率

**解码延迟（图 4）**：
- 128K 上下文：OmniKV 实现 1.68 倍加速（21.0 tokens/s）
- 450K 上下文：使用单张 A100，OmniKV 实现 1.87 倍加速（相比 3 张 A100 的原始模型）
- 70% 内存节省下，Llama-3-8B 在单张 A100 上以 7.5 tokens/s 运行 450K 上下文
- 80% 内存节省下，Llama-3.1-70B 在单张 A100 上以 4.5 tokens/s 运行 150K 上下文

**Prefill 延迟**：
- OmniKV 的 prefill 延迟与全注意力相当（offload 被全注意力计算覆盖）
- InfLLM 虽然在 450K 上下文时 prefill 更快，但可能影响性能

#### Token 预算权衡（图 5）

| Token 预算 | 平均分 | 延迟 (ms) |
|-----------|--------|----------|
| 128 | 35.9 | 50.4 |
| 1024 | 36.9 | 62.5 |
| 6400 (30% Mem) | 37.4 | 135.1 |

- 即使仅选择 128 个 token 作为稀疏注意力上下文，平均分（35.9）仍高于 H2O+（35.2）
- 1024 token 在性能与延迟之间取得良好平衡

#### Needle-in-a-Haystack 测试（图 7）
- 在 Llama-3-8B-1M 上，OmniKV 在 512K 上下文下实现完美检索
- 在 Llama-3-8B-1048K 上，512K 上下文下也取得完美结果

#### 过滤层分析（图 6）
- **任务无关性**：不同任务的过滤能力趋势一致，表明过滤能力是模型固有特性
- **命中率**：过滤能力更强的层能更好地识别包含答案的上下文
- **性能相关性**：过滤能力与 LongBench 上的性能正相关

#### 更大模型验证（表 9）
- Llama-3.1-405B：OmniKV 在 Qasper（48.5 vs 50.0）和 Qmsum（25.9 vs 25.5）上表现接近原始模型

#### 与推理引擎的兼容性（表 10、11）

**LightLLM + OmniKV（tp=4）**：

| 设置 | OmniKV | 原始 | vLLM |
|------|--------|------|------|
| 128K, bs=16 | 46.2 ms/token | 73.5 ms | 72.3 ms |
| 256K, bs=8 | 44.9 ms/token | 75.4 ms | 73.1 ms |
| 512K, bs=4 | 44.9 ms/token | 78.1 ms | 75.6 ms |

- OmniKV 在张量并行（TP）下仍能实现显著加速
- 兼容 Continuous Batching、Tensor Parallelism、Pipeline Parallelism、Context Parallelism

---

## 优势

1. **零性能损失的加速**：1.68 倍解码加速且不丢弃任何 token，保持与原始模型几乎一致的性能。
2. **无需训练**：完全基于推理阶段的注意力分数动态选择，无需额外训练或微调。
3. **极高的内存效率**：KV cache 内存占用降低 75%，将单张 A100 支持的最大上下文从 128K 扩展到 450K。
4. **与 offloading 高度兼容**：通过将大部分层的 KV cache 卸载到 CPU，仅在需要时加载少量 token，大幅减少 PCIe 传输量。
5. **简单易实现**：仅需对 Huggingface Transformers 做少量修改即可实现，代码开源。
6. **与现有技术正交**：可与 KV cache 量化（KIVI、SmoothQuant）、Flash Attention 等方法结合使用。
7. **多步推理场景优势显著**：在 CoT 等多步推理场景中表现优于所有基线。
8. **支持多种并行策略**：兼容 Tensor Parallelism、Pipeline Parallelism、Context Parallelism，适配 LightLLM 等推理引擎。
9. **Prefill 加速**：作者还扩展了 OmniKV 用于 prefill 加速，实现 1.90 倍延迟降低（256K 输入）。
10. **任务无关的过滤层选择**：过滤层的过滤能力是模型固有特性，一次确定后可适用于所有任务。

---

## 局限

1. **Prefill 阶段加速有限**：虽然论文扩展了 prefill 加速（OmniKV-prefill），但与专用 prefill 加速方法 MInference 相比仍有差距（256K: 107.2s vs 61.6s）。
2. **依赖手动设置过滤层超参数**：过滤层 L 需要手动确定，虽然任务无关，但仍需要在不同模型上进行调优。
3. **CPU-GPU 通信开销**：虽然通过 Packed Load 减少了传输次数，但在极端长上下文场景下，GPU-CPU 通信仍可能成为瓶颈。
4. **仅测试了有限的模型规模**：虽然验证了 8B、9B、70B、405B 模型，但未覆盖更多架构和规模。
5. **Single-Step 推理中与 Full Attention 接近但并非完全一致**：在某些任务上（如代码相关任务）OmniKV 性能可能略低于原始模型。
6. **2StageRetr 数据集规模有限**：平均长度仅 739 token，最大 1382 token，可能不足以全面评估长上下文能力。
7. **未与 Flash Attention 2 等最新优化进行对比**：主要基于 Huggingface Transformers 实现，可能未完全发挥硬件性能。
8. **重复输出问题的缓解**：对于摘要任务使用 top-p 采样（p=0.95, temperature=0.8），但这可能引入额外的性能差异。

---

## 与 EfficientPaper 相关的研究方向

### 关键词关联：`kv_cache_sparse`

OmniKV 属于 EfficientPaper 项目中 **KV cache 稀疏化** 方向的研究，与以下研究方向密切相关：

1. **KV Cache 压缩与剪枝**
   - 与 H2O（注意力分数丢弃）、SnapKV、ScissorHands 等方法互补
   - OmniKV 的独特之处在于不丢弃 token，仅动态选择上下文子集

2. **KV Cache Offloading（卸载）**
   - 与 FlexGen、InfLLM、Brutal Offload 等方法相关
   - OmniKV 通过稀疏注意力大幅减少 CPU-GPU 传输量

3. **注意力稀疏性与动态选择**
   - 与 Minference（prefill 阶段稀疏注意力）、Quest、SparQ、InfiniGen 等方法相关
   - OmniKV 首次揭示层间注意力相似性，减少稀疏模式的计算开销

4. **长上下文推理优化**
   - 与 LongLoRA、YaRN 等上下文扩展方法互补
   - OmniKV 使单张 A100 支持 450K 上下文

5. **KV Cache 量化**
   - 与 KIVI、SmoothQuant 等量化方法正交，可结合使用
   - 未来工作方向：OmniKV + KV cache 量化进一步提升效率

6. **推理引擎优化**
   - 与 vLLM、LightLLM 等推理引擎兼容
   - 支持 Continuous Batching、Tensor Parallelism、Pipeline Parallelism、Context Parallelism

### 核心研究价值
OmniKV 提出了一个新的 KV cache 优化范式：**不丢弃 token，而是动态选择上下文子集**。这一范式在多步推理场景中具有显著优势，且通过层间注意力相似性实现了高效的稀疏注意力计算。该方法为长上下文 LLM 推理提供了新的思路，与现有技术高度互补，具有很强的实用价值。

---

*本 note 由 AI Agent（Hermes Agent）自动生成。*
*生成时间：2025年6月4日*
*论文来源：ICLR 2025*
*论文 URL：https://openreview.net/forum?id=ulCAPXYXfa*
*代码仓库：https://github.com/antgroup/OmniKV*
