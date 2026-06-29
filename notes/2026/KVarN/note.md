---
title: KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks
arxiv: 2606.03458
url: http://arxiv.org/abs/2606.03458v1
date: 2026-06-02
---

# KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks

**本文由 AI Agent 自动生成，仅供研究参考，不代表原作者观点。**

## Abstract

Test-time scaling is a powerful approach to obtain better reasoning in large language models, but it becomes memory-bottlenecked during long-horizon decoding, as the KV-cache grows. KV-cache quantization can help improve this, but current methods are evaluated under prefill-like settings and errors behave differently under autoregressive decoding. We show that in the latter regime, quantization errors accumulate across timesteps, driven primarily by incorrect token scales. We introduce KVarN, a calibration-free KV-cache quantizer that applies a Hadamard rotation followed by a dual-scaling variance normalization across both axes of the K and V matrices. We find that this combination fixes outlying token-scale errors and substantially reduces error accumulation over existing baselines. KVarN establishes a new state-of-the-art for KV-cache quantization on generative benchmarks, including MATH500, AIME24 and HumanEval, at 2-bit precision. A vLLM implementation of the KVarN method is available at https://github.com/huawei-csl/KVarN.

## 一句话总结

KVarN 是一种无需校准的 KV-cache 量化方法，通过 Hadamard 旋转和双尺度方差归一化，显著减少自回归解码中的量化误差累积，在 2-bit 精度下在 MATH500、AIME24 和 HumanEval 等基准测试上达到了最先进的水平。

## 背景与问题

### 1. Test-time Scaling 导致 KV-cache 内存瓶颈
- Test-time scaling 是提升 LLM 推理能力的强大方法，但随着解码长度增加，KV-cache 增长导致内存瓶颈
- KV-cache 量化（2-4 bit）是解决内存瓶颈的有效途径
- **关键问题**：现有方法主要在 prefill-like 设置下评估，未考虑自回归解码中误差的累积效应

### 2. 量化误差在自回归解码中累积
- 在自回归解码中，量化误差会跨时间步累积（图4所示）
- Transformer 块 Bl 的注意力使用量化 KV-cache 计算，误差影响 Bl+1 的 K/V 矩阵
- 这种累积效应在长序列生成（如推理任务）中尤为严重

### 3. 误差累积的主要驱动因素
- **Magnitude Error（幅度误差）**是主要驱动因素（图1a）
- 错误的 token scale 是幅度误差的主要原因
- 前5%的异常误差对端到端性能的影响远大于其余95%（图3）

## 核心方法

### 1. KVarN 方法概述
KVarN 是一种**无需校准**的 KV-cache 量化方法，包含两个关键变换：

**Step 1: Hadamard 旋转（通道维度）**
- 在通道维度应用 Hadamard 旋转，减少通道异常值
- 遵循 QuaRot 的布局（图7）
- 降低通道空间中的异常值（incoherence processing）

**Step 2: 双尺度方差归一化（通道+Token维度）**
- 在通道和 Token 两个维度应用方差归一化（VarN()）
- 通过迭代归一化行/列方差实现
- 额外的元素级缩放只增加每个 token 每通道一个 FLOP 的开销
- 修复了离群 token-scale 错误，抑制了误差累积

### 2. 量化误差分解（Magnitude vs Direction）
- **总误差** = 幅度误差 (EM) + 方向误差 (ED)
- **幅度误差** = (∥K∥ - ∥Kdq∥)²
- **方向误差** = 2∥K∥∥Kdq∥(1 - cos θ)
- 量化异常误差主要由错误的 token 幅度驱动（图1a）

### 3. 伪解码评估方法（Pseudo-decode）
- 将完整 prefill 序列分成大小为 b 的块
- 每处理 b 个 token 后量化 KV-cache
- 后续块访问量化后的缓存，误差累积随时间增长
- 这种设置更好地模拟了长序列解码中的误差累积效应

### 4. 双尺度方差归一化原理
- 与权重量化中的双尺度缩放类似，但在 KV-cache 量化中有不同的原因（无校准数据）
- 迭代归一化列/行方差，使矩阵行列方差均匀
- 直接减少矩阵重构误差（非近似校准数据）
- 有效减少尾部误差（主要来自错误的 token 缩放）

## 技术细节

### 1. 量化设置
- **基础 tile**：(head-dim × token-chunk)，如 128 × 128（Llama3.1-8B）
- **K 矩阵**：per-channel 量化（KIVI 方案）
- **V 矩阵**：per-token 量化
- **精度**：2-bit（平均 2.3 bits/elem）
- **量化方式**：Round-to-nearest (RTN)

### 2. 方差归一化实现
- 基于 SINQ [22] 的 log-domain standard-deviation-scaling 实现
- 迭代归一化行/列方差（而非直接归一化 magnitude）
- 避免增加 per-channel kurtosis
- 每 128 tokens 块进行一次

### 3. 开销分析
- **归一化开销**：1.9 ms（128 token 生成）vs 1050 ms（标准解码），仅 0.18% 开销
- **解量化开销**：~1%（比 RTN 慢，但比 codebook 方法快）
- 对于更大模型，相对开销更低

## 实验设置

### 模型
- Qwen3-4B（原生支持推理）
- Llama-3.1-8B（无推理变体）
- Phi-4-14B（有推理变体）

### 基准测试
- **MATH500**：数学推理，复杂推导和数学解
- **AIME24**：竞赛级数学能力，长链思考
- **HumanEval**：编程能力，自然语言到 Python 代码
- **IFEval**：指令遵循，格式和内容约束
- **Line-Retrieval**：比 NiaH 更具信息量

### 基线方法
- **KIVI**：per-channel K / per-token V 量化（2-bit）
- **QuaRot**：Hadamard 旋转（2-bit）
- **KVQuant-1%**：非均匀量化 + 异常值处理（2-bit）
- **PolarQuant**：4-bit/2-bit 混合精度
- **TurboQuant**：codebook 方法（3-bit/3-bit）
- **Kitty**：2-bit + 通道重要性选择（2.4 bits）

### 评估设置
- **Prefill-like**：传统设置，不考虑误差累积
- **Pseudo-decode**：新设置，模拟自回归解码中的误差累积
- **端到端评估**：在生成式基准测试上评估（非合成任务）

## 主要结果

### 1. 端到端推理和指令遵循性能（表1-3）
- **KVarN 在 2-bit 精度下达到 SOTA**（平均 2.3 bits/elem）
- 在 MATH500、AIME24、HumanEval 和 IFEval 上优于或匹配所有基线

#### Qwen3-4B 结果
- **AIME24**：60.0%（vs KIVI 55.5%，QuaRot 56.7%）
- **MATH500**：79.2%（vs KIVI 77.8%，QuaRot 78.9%）
- **HumanEval**：88.4%（vs KIVI 86.4%，QuaRot 86.3%）
- **IFEval Strict**：80.4%（vs KIVI 80.3%，QuaRot 79.3%）

#### Phi-4-14B 结果
- **AIME24**：61.7%（vs KIVI 57.8%，QuaRot 58.9%）
- **MATH500**：84.8%（vs KIVI 74.4%，QuaRot 77.0%）
- **HumanEval**：88.2%（vs KIVI 74.6%，QuaRot 87.0%）
- **IFEval Strict**：63.4%（vs KIVI 60.6%，QuaRot 62.6%）

### 2. 误差累积抑制效果（图5）
- KVarN 在所有上下文长度上误差低于 KIVI
- 误差随上下文增长的累积速度显著降低
- 在长上下文中优势更加明显

### 3. 幅度误差修复效果（图1b）
- KVarN 有效抑制了幅度误差
- 防止了四舍五入过程中 worst-case tokens 的范数缩放
- 与 Hadamard 旋转协同作用

### 4. Line-Retrieval 准确率（表4）
- KVarN 在所有模型和基线上表现最佳

## 优点与局限

### 优点
1. **无需校准**：不需要校准数据，简化部署
2. **极低开销**：归一化仅 0.18% 开销，解量化 ~1% 开销
3. **统一精度**：uniform precision（2-bit），避免混合精度的复杂性
4. **显著改进**：在推理和编程任务上显著优于现有方法
5. **误差累积抑制**：有效解决自回归解码中的误差累积问题
6. **开源实现**：提供 vLLM 实现（https://github.com/huawei-csl/KVarN）

### 局限
1. **单 GPU 评估**：主要在单 GPU 上评估，未考虑分布式推理
2. **模型规模**：主要在 4B-14B 规模模型上评估
3. **比特精度限制**：仅在 2-bit 精度下评估，未探索更低精度
4. **理论分析**：缺少误差累积的严格理论分析
5. **任务范围**：主要关注推理和编程任务，未涉及其他任务（如翻译、摘要）

## 与 EfficientPaper 主题的关系

本文与 EfficientPaper 的核心主题高度相关：

1. **KV-cache 量化**：KV-cache 量化是 LLM 推理优化的重要方向，与 EfficientPaper 关注的高效推理一致
2. **内存优化**：通过 2-bit 量化显著减少 KV-cache 内存占用
3. **推理效率**：在 test-time scaling 场景下提升推理效率
4. **Error Accumulation**：关注自回归解码中的误差累积问题，是量化领域的重要问题

### 相关论文
- **KIVI** (2024): 基础 KV-cache 量化方法
- **QuaRot** (2024): Hadamard 旋转在量化中的应用
- **SINQ** (2022): 方差归一化在权重量化中的应用
- **TurboQuant** (2024): Codebook 量化方法

## 可复现/实现要点

### 1. 依赖
- vLLM（提供量化实现）
- PyTorch
- NumPy

### 2. 关键参数
- **Hadamard 旋转**：遵循 QuaRot 布局
- **方差归一化**：每 128 tokens 块进行
- **量化精度**：2-bit（K 和 V）
- **Tile 大小**：128 × 128

### 3. 实现步骤
1. 安装 vLLM
2. 克隆 KVarN 仓库
3. 运行量化脚本
4. 评估性能

### 4. 评估建议
- 使用 pseudo-decode 评估方法
- 在 MATH500、AIME24、HumanEval 等基准上评估
- 比较 KIVI、QuaRot 等基线

## 个人备注

- KVarN 的关键创新在于结合 Hadamard 旋转和双尺度方差归一化，有效解决了 KV-cache 量化中的误差累积问题
- 0.18% 的归一化开销非常低，使得方法实用
- 2-bit 精度在推理和编程任务上已经能达到接近 FP16 的性能，说明量化技术已经非常成熟
- 后续工作可以探索更低精度（1-bit）或更复杂的量化策略
- 与 Event Tensor (arXiv 2604.13327) 的对比：KVarN 专注于 KV-cache 量化，而 Event Tensor 专注于动态 megakernel 编译

---

*本文由 AI Agent 自动生成，仅供研究参考，不代表原作者观点。*
*生成时间：2026-06-02*
