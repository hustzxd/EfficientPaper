# H1B-KV: Hybrid One-Bit Caches for Memory-Efficient Large Language Model Inference

> Harshil Vejendla · Rutgers University

![111](../../blank.jpg)

---

> **⚠ 生成声明**：本 note 由 AI Agent（Hermes Agent）于 2026 年 6 月自动生成，基于 arXiv 论文原文（2510.05529v1）进行阅读、分析与中文撰写。内容仅供参考，可能与原文存在理解偏差。

---

## 一句话总结

H1B-KV 提出了一种混合 KV 缓存压缩方案：将 Key 用 1-bit 二值随机投影（binary sketch）表示，Value 用 4-bit 量化，通过轻量微调恢复模型性能，实现 7B 模型在 8k 上下文下仅需 ~59 MB 缓存（约 70 倍压缩），在内存、延迟和能效方面显著优于 KIVI、SparseLLM、Loki 等方法。

---

## 摘要翻译

大语言模型（LLM）的自回归解码需要缓存过去所有的 Key-Value（KV）对，使长上下文推理成为内存瓶颈问题。虽然近期方法探索了量化缓存、驱逐 token 或对 Key 使用二值草图（如 Loki），但这些方法通常不完整——要么 Value 未压缩，要么丢弃上下文信息。本文提出 Hybrid One-Bit KV Cache（H1B-KV），一种全面的压缩方案，能在不牺牲上下文的情况下大幅减少内存使用。H1B-KV 用 1-bit 二值草图表示每个 Key 向量，实现硬件友好的位运算注意力，并用 4-bit 量化压缩 Value 向量。这种整体混合方案使 7B 参数 LLM 在 8k token 上下文中仅需不到 60 MB 缓存——约 70 倍缩减。论文证明，经过轻量微调，H1B-KV 在困惑度基准以及数学推理（GSM8K）、多任务理解（MMLU）和代码生成（HumanEval）等复杂下游任务上匹配全精度性能，且显著优于 KIVI（量化）、SparseLLM（token 驱逐）和 Loki（仅 Key 草图）方法。

---

## 研究动机

1. **KV 缓存内存瓶颈**：LLM 自回归解码中，KV 缓存随序列长度线性增长。7B 模型在 32k token 上下文中，FP16 精度下缓存可超过 16 GB，使边缘设备（手机、嵌入式系统）部署不可行。

2. **现有方法的不足**：
   - **量化方法**（KIVI、MiniCache）：虽有效但仍存储多字节浮点/整数表示，内存和带宽占用仍显著。
   - **Token 驱逐方法**（SparseLLM、Keyformer）：丢弃"不重要"的 token，但对需要长距离依赖的任务（数学推理、代码生成）可能导致灾难性失败。
   - **Key 侧草图方法**（Loki）：仅压缩 Key 为 1-bit，Value 仍为 FP16，价值缓存仍占原始内存 50%，属于不完整方案。

3. **核心问题**：能否用极端的二值草图替换高维 Key 向量，同时压缩 Value 向量，而不损害模型性能？

---

## 方法（技术细节）

### 1. 理论基础：局部敏感哈希（LSH）

- 设 q, k ∈ R^d 为单位范数向量（query 和 key 通常经过层归一化）。
- 使用固定随机矩阵 R ∈ R^{b×d}（b < d，b 为草图宽度，如 256），R_ij ~ N(0,1)。
- 二值草图：s_q, s_k ∈ {−1,1}^b，由 s = sign(Rv) 生成。

**核心命题（Proposition 1）**：两个二值草图的归一化 Hamming 内积的期望值为：

$$E\left[\frac{1}{b} s_q^T s_k\right] = 1 - \frac{2}{\pi} \arccos(q^T k)$$

即二值注意力分数是原始余弦相似度的有原则的非线性近似。通过微调 softmax 温度 T 来重新校准分数分布，恢复模型性能。

### 2. One-Bit Key Sketching

- 每个注意力头的 key 向量 k ∈ R^d 由固定随机矩阵 R 投影一次：s_k = sign(Rk)
- 存储仅需 b bits（如 256 bits = 32 bytes）
- 解码时，query q 用同一矩阵 R 投影得 s_q
- 注意力分数直接在二值空间计算：a_t = (1/b) s_q^T s_k
- 使用 XNOR 和 POPCOUNT 位运算指令高效实现

### 3. Hybrid Cache with Value Quantization

- 仅压缩 Key 不够，Value 向量仍是重要瓶颈
- 对 Value 使用 per-tensor 仿射量化：v_q = round(v/s + z)，其中 s 为 scale，z 为 zero-point
- 每个 token 存储：b-bit key sketch + d × 4-bit 量化 value 向量
- 压缩比极低：1-bit key + 4-bit value，整体约 98% 压缩

### 4. 轻量微调（Lightweight Finetuning）

- **不微调**原模型参数，只训练极少量参数（< 0.1%）：
  - 全局 softmax 温度标量 T
  - Value 的仿射投影层（V_proj）
- 注意力输出：Attention(q, K, V) = softmax(a_t / T) · V_dequant
- 60M 模型：约 2 小时（单 A100 GPU）
- 7B 模型：约 5 A100 小时
- 一次性成本，相对于推理时的长期节省可忽略

---

## 实验结果

### 主要对比（60M 模型，8k 上下文）

| 方法 | 缓存大小 (MB) | WikiText-2 PPL | QpB | PTB PPL | QpB |
|------|-------------|---------------|------|---------|------|
| FP16 | 67.1 | 9.20 | 1.63 | 5.98 | 2.50 |
| MiniCache 4-b | 16.8 | 9.42 | 6.30 | 6.10 | 10.15 |
| KIVI 2-b | 8.4 | 9.65 | 12.28 | 6.25 | 19.04 |
| SparseLLM | 8.4 | 9.58 | 12.44 | 6.20 | 19.22 |
| Loki (1-bit K) | 33.7 | 9.24 | 3.23 | 6.01 | 5.51 |
| **H1B-KV (Ours)** | **5.3** | **9.28** | **20.35** | **6.02** | **31.37** |

- H1B-KV 将 8k 上下文缓存从 67 MB 压缩至 5.3 MB（~12x 压缩）
- QpB（质量-字节比）比次优方法高 5-8 倍

### 7B 模型可扩展性

| 方法 | 缓存大小 (MB) | WikiText-2 PPL | QpB |
|------|-------------|---------------|------|
| FP16 | 4300.2 | 5.05 | 0.05 |
| KIVI 2-b | 537.5 | 5.30 | 0.35 |
| SparseLLM | 537.5 | 5.25 | 0.36 |
| Loki (1-bit K) | 2150.1 | 5.08 | 0.09 |
| **H1B-KV** | **58.7** | **5.15** | **3.31** |

- 从 4.3 GB 压缩至 58.7 MB（~73x 压缩）
- PPL 接近 FP16 基线

### 下游任务（7B 模型）

| 方法 | GSM8K (%) | MMLU (%) | HumanEval pass@1 (%) |
|------|----------|---------|---------------------|
| FP16 | 54.2 | 68.1 | 28.5 |
| KIVI 2-b | 52.8 | 67.5 | 27.1 |
| SparseLLM | 15.7 | 65.2 | 9.3 |
| **H1B-KV** | **53.5** | **67.9** | **28.1** |

- H1B-KV 与 FP16 差距极小
- SparseLLM 在 GSM8K 上崩溃（15.7% vs 54.2%），因驱逐 token 导致关键信息丢失

### 硬件延迟与能效（7B，8k 上下文）

- Raspberry Pi 5：2.8x 加速，能耗降低 60%
- NVIDIA Jetson Nano：2.1x 加速
- Cache Load 从 25ms 降至 3ms，Attention Compute 从 41ms 降至 11ms

### 消融实验

- **1-Bit-Key Only**（FP16 Values）：PPL 好但内存节省不够（33.8 MB）
- **无温度微调**：PPL 大幅劣化（12.51 vs 9.28），确认温度 T 的关键作用
- **草图宽度 b**：b=256 最佳平衡；b<64 时近似质量急剧下降

---

## 优势

1. **极端压缩**：98% 压缩率，7B 模型在 8k 上下文下仅需 ~59 MB
2. **完整上下文保留**：不同于 token 驱逐方法，H1B-KV 保留全部 token，避免信息丢失
3. **硬件友好**：位运算（XNOR + POPCOUNT）可高效在 CPU/GPU 上实现
4. **轻量微调**：仅需 < 0.1% 参数更新，一次性成本低
5. **广泛适用**：60M 到 7B 模型均可适配，理论基础扎实（LSH）
6. **下游任务表现优秀**：在 GSM8K、MMLU、HumanEval 上接近全精度性能
7. **边缘设备部署可行**：在 Raspberry Pi 5 和 Jetson Nano 上有显著加速和节能

---

## 局限

1. **草图宽度下限**：b < 64 时近似质量急剧下降，不适合极低维表示
2. **微调范围有限**：未联合优化随机投影矩阵 R 或量化参数，可能进一步提升性能
3. **多模态模型待探索**：未验证在具有不同数据分布的多模态模型上的适用性
4. **需要微调**：不同于 KIVI 的 tuning-free 方案，H1B-KV 需要轻量微调步骤
5. **仅单次投影**：随机矩阵 R 是预生成的，非学习得到

---

## 与 EfficientPaper 相关的研究方向

1. **KV Cache 量化**：H1B-KV 是 KV 缓存压缩的重要进展，可与 KIVI、MiniCache 等方法对比或结合使用。
2. **边缘推理部署**：该方法在 Raspberry Pi 5 和 Jetson Nano 上的实验证明其对边缘设备部署的价值。
3. **比特级压缩**：1-bit Key + 4-bit Value 的混合方案为 KV 缓存的极低比特压缩提供了新思路。
4. **轻量微调策略**：仅微调 < 0.1% 参数的方案对高效适配模型提供了参考。
5. **Local-Sensitive Hashing 在 LLM 中的应用**：将 LSH 理论应用于 Transformer 注意力机制，为稀疏/近似注意力研究提供理论基础。
6. **对比学习方向**：与 Loki（仅 Key 草图）、KIVI（2-bit 量化）、SparseLLM（token 驱逐）的对比分析有助于理解不同压缩策略的优劣。
7. **未来方向**：可学习投影矩阵、多模态模型扩展、更激进的 Value 压缩（如 2-bit）等。
