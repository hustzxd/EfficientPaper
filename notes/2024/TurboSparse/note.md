# Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters

![](../../blank.jpg)

## 一句话总结

TurboSparse 提出了一种新颖的 dReLU 激活函数，通过在 Gated-MLP 的 gate 和 up 两个投影层同时应用 ReLU，将 LLM 推理时的激活稀疏度从 40% 提升至 90%，仅激活 2.5B-4.3B 参数即可在 Mistral-7B 和 Mixtral-47B 上实现 2-5× 解码加速，并在移动端（OnePlus-12）以 11 tokens/s 速度运行 47B 参数模型。

---

## 摘要翻译

利用激活稀疏性是显著加速大语言模型（LLM）推理过程的有前景的方法，且不损害模型性能。然而，激活稀疏性由激活函数决定，常用的 SwiGLU 和 GeGLU 等激活函数表现出有限的稀疏性。简单地将这些函数替换为 ReLU 无法实现足够的稀疏性。此外，不充分的训练数据会进一步增加性能下降的风险。为解决这些挑战，我们提出了一种新颖的 dReLU 函数，旨在提高 LLM 激活稀疏性，并结合高质量的训练数据混合比例以促进有效的稀疏化。此外，我们利用混合专家（MoE）模型中前馈网络（FFN）专家的稀疏激活模式来进一步提升效率。通过将我们的神经元稀疏化方法应用于 Mistral 和 Mixtral 模型，每次推理迭代分别仅激活 25 亿和 43 亿参数，同时实现更强大的模型性能。评估结果表明，该稀疏性实现了 2-5 倍的解码加速。值得注意的是，在手机上，我们的 TurboSparse-Mixtral-47B 实现了 11 tokens/s 的推理速度。

---

## 研究动机

### 背景问题

当前主流 LLM（如 Llama、Mistral、Gemma）均为稠密模型，推理时使用全部参数。随着模型规模增长，计算资源需求急剧增加，成为 AI 广泛部署的主要障碍。

### 已有方法的局限

1. **MoE 方法**：通过专家路由实现条件计算，但架构设计复杂，且专家内部仍为稠密激活。
2. **ReLU 激活稀疏性**：ReLU 天然输出零元素，Deja Vu 可实现 2× 加速，PowerInfer 可达 11× 加速。但当前 LLM 通常使用 SwiGLU/GeGLU，其稀疏性有限（约 40%）。
3. **ReLUfication（已有最先进方法）**：将激活函数替换为 ReLU 后继续预训练，但存在两个关键缺陷：
   - 仅替换 gate 投影的 ReLU 效率有限，稀疏度仅从 40% 提升到约 67-71%（ReLULlama-7B）
   - 训练数据多样性不足、训练 token 数不够，导致能力恢复不完全

### 核心洞察

作者发现现有 ReLUfication 方法只关注修改 gate 投影，而忽略了 up 投影的激活分布。gate 和 up 投影共同影响神经元激活的稀疏性，up 投影中大量激活值小于 0，这些负值也可被屏蔽，从而引入更强的稀疏性而不牺牲非线性能力。

---

## 方法（技术细节）

### 1. dReLU 激活函数

**核心公式**：

```
Combined_dReLU(x) := max(0, xW_gate) * max(0, xW_up)
```

与标准 Gated-MLP 结构的区别：
- **标准 SwiGLU**：`Gate(x) := SiLU(xW_gate)`，`Combined(x) := Gate(x) * Up(x)`
- **ReLUfication (ReGLU)**：仅在 gate 上应用 ReLU，`Gate(x) := max(0, xW_gate)`
- **dReLU（本文方法）**：在 gate 和 up 两个投影上同时应用 ReLU，实现双重稀疏化

### 2. 小模型验证

使用 300M 参数的 decoder-only 架构，dReLU 和 SwiGLU 在 fineweb 数据集上预训练 5B token：
- dReLU 训练损失：3.154，验证 PPL：28.45
- SwiGLU 训练损失：3.146，验证 PPL：28.48
- 结论：dReLU 收敛能力与 SwiGLU 相当，且验证 PPL 略优

### 3. 稀疏度-性能权衡分析

采用 Top-k% 方法控制稀疏度（取绝对值最大的 k% 激活值）：

| Top-k% | dReLU PPL | SwiGLU PPL |
|--------|-----------|------------|
| 0%     | 28.45     | 28.48      |
| 50%    | 28.45     | 28.62      |
| 80%    | 28.45     | 36.28      |
| 85%    | 28.65     | 48.55      |
| 90%    | 29.19     | 112.36     |

关键发现：dReLU 在 90% 稀疏度下仍保持竞争力，而 SwiGLU 在 80% 时已严重退化。

### 4. MoE 模型中的稀疏性

对 Deepseek-MoE、Qwen1.5-MoE、Mixtral 进行分析：
- MoE 模型的 FFN 专家内部仍存在类似稠密模型的稀疏激活模式
- 稀疏度达 50% 时性能仅下降约 1-2%
- 这意味着 ReLUfication 可扩展到 MoE 模型，且 FFN 权重占比更高，FLOP 减少更显著

### 5. 大规模预训练

**模型**：Mistral-7B 和 Mixtral-47B（将 SwiGLU FFN 替换为 dReLU FFN）

**训练数据混合比例**（约 150B token，不到典型预训练 token 的 1%）：
- Web 数据（Wanjuan-CC 等）：约 74%
- 学术数据（Arxiv、PubMed、Philpapers 等）：约 10%
- 书籍数据：约 7%
- 数学数据：约 3%
- 代码数据（Starcoder、GitHub-Code）：约 6%

**超参数**：
- 序列长度：4096
- 批量大小：2048
- 学习率：5e-5 → 5e-6（余弦调度）
- 预热步数：1000
- 优化器：AdamW（β1=0.9, β2=0.95）
- 硬件：64 × A800-80G GPU

**SFT 阶段**：使用 orca-math-word-problems、bagel 等高质量 SFT 数据集，同时为每个 FFN 块训练预测器模块（TurboSparse-Mixtral-47B 为每个专家训练预测器），用于预测哪些神经元将被激活。

---

## 实验结果

### 1. 下游任务性能

| 模型 | 参数量 | 激活参数 | ARC-challenge | Hellaswag | MMLU | TruthfulQA | WinoGrande | GSM8k | OpenLLM Avg |
|------|--------|---------|---------------|-----------|------|------------|------------|-------|-------------|
| Gemma-2B | 2B | 2B | 48.55 | 71.02 | 40.05 | 34.38 | 66.06 | 18.72 | 46.46 |
| Mistral-7B | 7B | 7B | 61.43 | 83.32 | 62.65 | 44.06 | 79.24 | 40.17 | 61.57 |
| **TurboSparse-Mistral-7B** | 7B | **2.5B** | 62.20 | 82.17 | 63.89 | 46.64 | 76.16 | 50.84 | **63.65** |
| Mixtral-47B | 47B | 13B | 68.09 | 86.62 | 70.53 | 48.59 | 83.35 | 58.91 | 69.34 |
| **TurboSparse-Mixtral-47B** | 47B | **4.3B** | 67.49 | 85.22 | 70.48 | 56.64 | 82.24 | 68.50 | **71.76** |

关键发现：
- TurboSparse-Mistral-7B 仅激活 2.5B 参数，性能优于 Gemma-2B（全参数 2B），且超越原始 Mistral-7B
- TurboSparse-Mixtral-47B 仅激活 4.3B 参数，性能超越原始 Mixtral-47B（激活 13B 参数）
- 在 GSM8k 上提升尤为显著（Mistral: 50.84 vs 40.17, Mixtral: 68.50 vs 58.91）

### 2. 稀疏度

- **TurboSparse-Mistral-7B**：平均每层 90% 神经元不激活
- **TurboSparse-Mixtral-47B**：每个专家 FFN 平均 85% 不激活；结合 MoE 路由的 75% 稀疏，每个 MoE 层仅激活 3% 的参数

### 3. 推理加速

**纯 CPU 推理（tokens/s）**：

| 硬件 | 模型 | PowerInfer | llama.cpp | 加速比 |
|------|------|-----------|-----------|--------|
| PC-2080Ti | Mistral-7B-FP16 | 9.94 | 4.78 | 2.08× |
| PC-2080Ti | Mixtral-47B-INT4 | 11.98 | 4.26 | 2.81× |
| PC-Laptop | Mistral-7B-FP16 | 8.71 | 4.13 | 2.11× |
| PC-Laptop | Mixtral-47B-INT4 | 16.1 | 6.91 | 2.32× |

**CPU/GPU 混合推理（tokens/s）**：

| 硬件 | 模型 | PowerInfer | llama.cpp | 加速比 |
|------|------|-----------|-----------|--------|
| PC-2080Ti | Mistral-7B-FP16 | 35.5 | 7.64 | **4.64×** |
| PC-2080Ti | Mixtral-47B-INT4 | 22.24 | 6.63 | 3.35× |
| PC-Laptop | Mixtral-47B-INT4 | 33.12 | 13.1 | 2.52× |

**移动端推理（OnePlus-12）**：

| 模型 | PowerInfer-2 | llama.cpp | 加速比 |
|------|-------------|-----------|--------|
| Mixtral-47B-INT4 | 11.1 tokens/s | 0.5 tokens/s | **22.2×** |

平均生成加速约 2.83×。

---

## 优势

1. **极高的激活稀疏度**：通过 dReLU 将稀疏度从 40% 提升至 90%，MoE 模型结合路由稀疏可达 97%
2. **性能不降反升**：TurboSparse 模型在多项基准上超越原始模型，特别是在 GSM8k（数学推理）和 TruthfulQA 上
3. **训练效率高**：仅需约 150B token（不到典型预训练的 1%），大幅降低训练成本
4. **通用性强**：可同时应用于稠密模型（Mistral-7B）和 MoE 模型（Mixtral-47B）
5. **实际部署价值**：在移动端实现 22.2× 加速（Mixtral-47B），使大模型在移动设备上可用
6. **方法简洁**：dReLU 仅修改激活函数（公式简洁），无需复杂的架构改动

---

## 局限

1. **训练成本仍存在**：虽仅需 150B token，但仍需 64 × A800-80G GPU，对个人开发者有门槛
2. **稀疏预测器的额外开销**：SFT 阶段需训练每个 FFN 块的预测器，MoE 模型还需为每个专家训练预测器
3. **对 ReLU 类激活函数的依赖**：方法依赖 ReLU 系列激活函数的稀疏特性，对其他激活函数（如 GELU）可能不适用
4. **实验模型规模有限**：主要在 7B 和 47B 规模验证，未在更大规模（如 70B、405B）上测试
5. **移动端验证有限**：仅在 OnePlus-12（Snapdragon 8 Gen 3）上测试，未覆盖更多移动平台
6. **与量化方法的联合效果未充分探索**：虽然使用了 INT4 量化进行速度对比，但与 dReLU 稀疏化的联合优化未深入研究

---

## 与 EfficientPaper 相关的研究方向

TurboSparse 与 EfficientPaper 项目中的多个研究方向密切相关：

1. **激活稀疏性（Activation Sparsity）**：本文的核心研究方向，通过 dReLU 函数实现高稀疏度，与 PowerInfer、Deja Vu 等工作属于同一研究线
2. **稀疏剪枝（Sparse Pruning）**：本文方法本质上是一种结构化稀疏方法，通过稀疏化激活函数实现高效推理
3. **混合专家模型（MoE）**：将稀疏激活扩展到 MoE 模型，发现 MoE 专家内部仍存在稀疏性，为 MoE 模型的进一步高效化提供了新思路
4. **高效推理（Efficient Inference）**：通过稀疏激活显著降低推理 FLOPs，与模型压缩（量化、蒸馏）、推测解码等方法互补
5. **移动端部署（Mobile Deployment）**：在移动设备上实现高速推理，与端侧 LLM 部署研究方向密切相关
6. **条件计算（Conditional Computing）**：通过激活稀疏性实现条件计算，是高效 AI 研究的重要方向

---

## AI 生成声明

本笔记由 AI Agent（Hermes Agent）自动生成，基于论文 PDF 文本提取和元数据信息。笔记内容经过整理和翻译，旨在提供论文的核心信息概览。如有需要，请参考原始论文获取完整细节。

---

*笔记生成时间：2026-06-05*
*论文链接：[arXiv:2406.05955v2](http://arxiv.org/abs/2406.05955v2)*
*代码仓库：[PowerInfer](https://huggingface.co/PowerInfer)*
