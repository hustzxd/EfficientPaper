# Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach

> Yaoxin Yang, Peng Ye, Xudong Tan, Chongjun Tu, Maosen Zhao, Jia Hao, Tao Chen

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

多模态大语言模型（MLLM）的推理开销很大，因为多模态 KV Cache 随视觉输入长度成比例增长。现有压缩方法主要依赖注意力分数来缩减缓存大小，这与 FlashAttention 等高效注意力内核不兼容，且忽略了 Value 向量对注意力输出的贡献。

本文从 KV 矩阵分布的角度重新审视多模态 KV Cache 压缩。首先发现多模态 KV 矩阵的频域能量主要集中在低频；进一步发现，移除偏离主能量较大的 KV 对会导致显著性能下降，定义为 Outlier KV。提出 FlashCache，一个频域引导的、Outlier-KV 感知的压缩框架，在多个 MLLM 和基准上实现最高 1.69x 解码加速、80% KV 内存降低，同时保持任务性能。

## 一句话总结

通过 DCT 频域分析发现多模态 KV 矩阵的低频能量集中特性，定义偏离主能量的 Outlier KV 为关键信息载体，并设计 Outlier KV 识别 + 动态预算分配模块，在不依赖注意力分数的情况下实现高效的多模态 KV Cache 压缩。

## 背景与问题

### 多模态 KV Cache 的瓶颈

MLLM 在长上下文多图/高分辨率/视频场景下，视觉 token 数量可达文本 token 的数百甚至数千倍，导致：
- KV Cache 内存急剧膨胀
- 解码阶段显著减速

### 现有方法的局限

现有 KV Cache 压缩方法（LOOK-M、MEDA 等）依赖注意力分数：
1. **与 FlashAttention 不兼容**：FlashAttention 不显式输出完整注意力矩阵，重计算引入额外开销
2. **忽略 Value 贡献**：注意力分数仅反映 Q-K 交互，忽略了 V 矩阵对输出的信息贡献

**核心问题**：如何在不依赖注意力分数的情况下进行高效的多模态 KV Cache 压缩？

## 核心方法

### 关键发现

**发现 1：频域低频集中**。对 KV 矩阵做 DCT 变换后，频域能量主要集中在低频分量，高频分量占比很小。

**发现 2：Outlier KV 假说**。用低通滤波提取主能量（Base KV）后，计算每个 KV 对与 Base KV 的偏差。实验证明：
- 移除**偏差大**的 KV 对 → 性能急剧下降
- 移除**偏差小**的 KV 对 → 性能影响较小
- 说明偏差大的 KV 对（Outlier KV）编码了推理关键特征

### FlashCache 框架

FlashCache 在 prefill 阶段结束后一次性压缩 KV Cache，包含两个核心模块：

#### 1. Outlier KV Recognition Module

对每层的 KV 矩阵执行：
1. **DCT 变换**：将 KV 矩阵映射到频域
2. **低通滤波**：保留低频分量（截断因子 γ=0.1~0.2），丢弃高频
3. **IDCT 逆变换**：回到时域，得到平滑的 Base KV
4. **偏差计算**：对每个 KV 对计算与 Base KV 的 MSE 偏差（分别计算 K 和 V 的偏差）
5. **Top-R 选择**：按偏差得分排序，保留偏差最大的 R 个 KV 对（Outlier KV）

#### 2. Dynamic Budget Allocation Module

观察到不同层的低频集中程度不同，因此动态分配每层的 KV Cache 预算：
1. 对每层计算频域中 outlier 信息能量与总能量的比值
2. 将比值归一化为权重
3. 在全局缓存预算下，为每层分配不同的保留配额

低频集中度高的层 → outlier 信息少 → 分配较少预算
低频集中度低的层 → outlier 信息多 → 分配较多预算

## 技术细节

### DCT 与低通滤波

- 使用离散余弦变换（DCT）将 KV 矩阵映射到频域
- 截断因子 γ 控制低通滤波的截止频率（γ 越小保留越多低频）
- 实验最优 γ 范围：0.1~0.2
- 使用 NVIDIA CuPy 加速 DCT 运算

### 偏差得分

对第 l 层的第 i 个 KV 对：
- DEV_k = MSE(K_i, K_base_i)（Key 偏差）
- DEV_v = MSE(V_i, V_base_i)（Value 偏差）
- 综合偏差得分用于排序选择 Outlier KV

### 压缩时机

在 prefill 阶段结束后一次性执行压缩，不影响解码阶段的正常流程。压缩后的 KV Cache 直接兼容 FlashAttention。

## 实验设置

### 模型
- LLaVA-OneVision-1.5-8B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-32B-Instruct

### 硬件
- 单卡 NVIDIA H200 (141GB)
- 使用 FlashAttention

### 基准
- **多图理解**：MileBench, MUIRBench, MMMU
- **高分辨率**：V*, HR-Bench
- **视频**：FAVOR-Bench

### Baseline
- StreamingLLM, H2O, SnapKV, LOOK-M, MEDA

### 配置
- KV Cache 保留率：ρ = 0.8, 0.6, 0.4, 0.2, 0.1, 0.05
- 重点评估 ρ = 0.2（高压缩）和 ρ = 0.1/0.05（极高压缩）

## 主要结果

### 多图理解（MileBench, ρ=0.2）
- FlashCache 在多数任务上优于所有 baseline
- 在 Needle-In-A-Haystack 任务上优势尤其显著（保留检索关键 KV）

### 极低保留率（ρ=0.05）
- 其他方法在 MUIRBench 上 OOM，FlashCache 正常运行
- 在 V* 上 FlashCache 达到 79.66%（vs Full Cache 80.23%），仅 0.57% 差距

### 高分辨率场景
- ρ=0.1 时 FlashCache 在 V* 上达到 80.23%（与 Full Cache 持平）

### 解码加速
- ρ=0.2 时最高 **1.69x** 解码加速
- 长序列下加速效果更显著（64K input 时延迟几乎恒定）
- KV 内存降低 **80%**

### 方法开销对比
FlashCache 额外开销远低于基于注意力的方法：
- 2K input：1.66ms（vs MEDA 16.6ms, LOOK-M 6.93ms）
- 8K input：6.77ms（vs MEDA 83.75ms, LOOK-M 53.97ms）

### 消融实验
- **截断因子 γ**：0.1~0.2 最优，过大会丢失 Base KV 主成分
- **动态预算分配（DBA）**：移除后性能明显下降（INIAH: 29.69→24.69, CLEVR: 41.04→35.85）

## 优点与局限

### 优点
1. **无需注意力分数**：与 FlashAttention 完全兼容，无需重计算注意力矩阵
2. **训练无关**：纯推理时压缩，无需微调或校准
3. **低开销**：DCT + MSE 的计算开销远低于注意力计算
4. **考虑 Value 贡献**：同时计算 K 和 V 的偏差，而非仅看 Q-K 交互
5. **动态预算分配**：自适应不同层的冗余程度
6. **鲁棒性好**：在极低保留率下性能衰减缓慢

### 局限
1. **仅验证 MLLM**：未在纯文本 LLM 上测试（文本 KV 的频域特性可能不同）
2. **DCT 开销**：虽然用 CuPy 加速，但 DCT 本身仍有额外计算（64K+ 输入时可能显著）
3. **单次压缩**：只在 prefill 后压缩一次，不支持解码阶段的动态调整
4. **截断因子 γ 需调优**：虽然 0.1~0.2 通用，但不同模型/任务可能需要不同值
5. **代码未开源**

## 与 EfficientPaper 主题的关系

FlashCache 属于 **KV Cache 稀疏化**（kv_cache_sparse）和 **KV Cache 管理**（kv_cache_management）交叉领域。核心创新点：

- **频域视角**：首次从频域分析多模态 KV 矩阵的分布特征
- **注意力无关**：不依赖注意力分数进行 KV 选择，开辟了新的压缩范式
- **多模态特化**：针对多模态场景中视觉 token 的特殊冗余模式

与 EfficientPaper 中的 H2O、SnapKV 等注意力分数方法形成互补。

## 可复现/实现要点

1. **DCT 实现**：使用 NVIDIA CuPy 加速
2. **关键参数**：截断因子 γ=0.1~0.2
3. **压缩流程**：prefill 后一次性执行 DCT→滤波→IDCT→MSE→排序→选择
4. **兼容性**：压缩后直接用 FlashAttention，无需特殊 kernel

## 个人备注

- 频域分析视角新颖，但 DCT 是否是最优的频域变换值得探讨（小波变换？）
- Outlier KV 的定义类似于量化中的 outlier 概念，两者可以结合
- 在纯文本 LLM 上的效果有待验证——文本 KV 的频域分布是否也呈现低频集中？
- 与 ContiguousChunk 思路的对比：FlashCache 是 token 级选择，ContiguousKV 是 chunk 级
