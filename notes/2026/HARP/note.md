# HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization

> Artur Zagitov, Gleb Molodtsov, Aleksandr Beznosikov

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Post-training quantization (PTQ) 是在内存和带宽受限环境中部署 LLM 的关键技术，但极低比特量化对 activation outliers 和 anisotropic weight curvature 非常敏感。已有 incoherence-based PTQ 方法通常使用固定 randomized Hadamard transforms (RHTs) 来混合坐标、降低 outlier 对量化的影响；RHT 高效、正交、易部署，但不能根据 layer、calibration distribution 或 downstream quantizer 自适应调整旋转基。

HARP（Hadamard-preconditioned Adaptive Rotation Processor）提出一个可学习的 structured two-sided orthogonal processor，用来替换固定 Hadamard/RHT mixing，同时保持 full-precision computation 等价。它把每个 rotation 表示为 sparse butterfly-like block-orthogonal stages 的乘积，支持 Mixed-Radix schedule 处理非 2 的幂维度，并初始化为与 RHT 等价的处理器，再用 calibration data 学到 layer/backend-specific 的结构化 refinement。实验显示，在 1B 到 70B 模型的 2–4 bit 设置中，HARP 相对固定 RHT 改善 perplexity 和 zero-shot accuracy；同时保留大部分部署效率，例如 2-bit Llama 2 7B 达到 128 tok/s，而 FP16 为 61 tok/s。

## 一句话总结

HARP 的核心是把 QuIP#/QTIP 这类 Hadamard-based PTQ 中“固定随机旋转”替换成“从 calibration data 学出来的结构化正交旋转”：仍然是 exact change of basis、仍然接近 Hadamard 的 `O(d log d)` 成本，但能更好适配每层曲率、权重分布和 blockwise quantizer。

## 背景与问题

LLM 推理常受内存带宽限制，PTQ 通过把权重和激活压到低比特来降低带宽与存储压力。进入 2–4 bit 极低比特区域后，量化误差高度受以下因素影响：

- 权重/激活中的 heavy-tailed outliers；
- Hessian 或 second-moment curvature 的 anisotropy；
- blockwise / vector quantizer 的固定分组结构；
- 量化 codebook 是否能有效表示旋转后的权重块。

QuIP、QuIP#、QTIP 等 incoherence processing 方法的基本思想是：在量化前做正交 change of basis，把 weight mass 和 curvature-sensitive directions 从少数坐标轴上打散，降低 blockwise quantization 的难度。随机 Hadamard transform (RHT) 很受欢迎，因为它正交、快速、支持 `O(d log d)` kernel。

但固定 RHT 的局限也很明显：它是通用随机混合，不知道某一层的 calibration distribution，也不知道下游 quantizer 的 block structure/codebook。论文提出的问题是：**能否学习一个 data-aware 的旋转，让它像 RHT 一样高效且保持 full-precision 等价，但比固定 RHT 更适合当前层和当前量化后端？**

## 核心方法

HARP 是一个 two-sided orthogonal processor。对于线性层 `W ∈ R^{d_out × d_in}` 和 calibration input second moment `H = E[xx^T]`，HARP 学习两个正交矩阵：

```text
U ∈ O(d_out),  V ∈ O(d_in)
```

量化前旋转：

```text
W~ = U^T W V
H~ = V^T H V
```

量化后映射回原空间：

```text
W_hat = U W~_hat V^T
```

由于 `U` 与 `V` 正交，full-precision computation 与 Hessian-weighted reconstruction objective 在数学上保持等价；改变的只是 quantizer 看到的坐标基。

HARP 的主要创新是如何参数化和学习 `U,V`：

1. 用 sparse stride stages / butterfly-like block-orthogonal factors 表示旋转，而不是 dense `d×d` 矩阵；
2. 每个 stage 由许多小 block-orthogonal kernels 组成；
3. block kernel 写成 `Q(θ) G_b`：`G_b` 是固定 Hadamard/QR base mixer，`Q(θ)` 是可学习正交 residual；
4. 初始化 `θ=0` 时，`Q=I`，整个 processor 恢复 RHT/QuIP# 的 Hadamard preprocessing；
5. 在 calibration data 上优化 `θ`，让旋转后的权重更容易被 downstream quantizer 表示。

## 技术细节

### 1. Layerwise PTQ objective

HARP 采用 PTQ 中常见的 layerwise reconstruction 视角。给定线性层权重 `W` 和 calibration inputs 的 second moment：

```text
H = E[x x^T]
```

量化目标可写为 Hessian-weighted output error：

```text
L(W, W_hat) = Tr((W - W_hat) H (W - W_hat)^T)
```

在正交变换后：

```text
W~ = U^T W V
H~ = V^T H V
```

目标等价为：

```text
L(W, W_hat) = Tr((W~ - W~_hat) H~ (W~ - W~_hat)^T)
```

因此，HARP 不改变 full-precision model，只改变有限 code quantizer 的工作坐标。

### 2. Structured orthogonal stages

dense learned rotation 存储和计算都是 `O(d^2)`，对 LLM 隐层维度不可行。HARP 使用 Mixed-Radix schedule：

```text
b = (b_0, ..., b_{m-1}),   Π_t b_t = d
```

每个 stage 只混合长度为 `b_t` 的小组。stage operator 可表示为：

```text
S_t(Θ_t) = P_t^T BlockDiag(B_{t,1}, ..., B_{t,D_t}) P_t
```

整个 transform 为：

```text
T(Θ) = S_{m-1}(Θ_{m-1}) ... S_0(Θ_0)
```

每个 block 都正交，因此整个 `T(Θ)` 正交。实际实现不显式 materialize permutation，而是通过 reshape/transpose 暴露 stride groups，再做 blockwise multiplication。

### 3. Hadamard-preconditioned block kernels

每个 block kernel 定义为：

```text
B_{t,c}(θ_{t,c}) = Q_{t,c}(θ_{t,c}) G_{b_t}
```

其中：

- `G_b` 是固定 base mixer；若 radix 为 2 的幂，使用 normalized Sylvester Hadamard；非 2 的幂时用固定 Gaussian matrix 的 QR 正交 fallback；
- `Q(θ)` 是可学习正交矩阵；
- `θ=0` 时 `Q=I`，所以 block 等于 base mixer。

这带来两个好处：

1. **强初始化**：训练开始时就是 RHT-like processor，而不是随机 dense rotation；
2. **局部 refinement**：学习的是围绕 Hadamard baseline 的结构化 residual，更适合非平滑的极低比特量化目标。

### 4. 正交参数化

- 对 `b=2`，使用 Givens rotation；
- 对 `b>2`，使用 Cayley map：

```text
Q(θ) = (I + A(θ))^{-1}(I - A(θ))
```

其中 `A(θ)` 是 skew-symmetric matrix。这样保证每个 block 始终正交。

### 5. Mixed-Radix 与非 2 的幂维度

HARP 支持非 power-of-two dimensions。例如 `5120` 可以用 schedule `(8, 8, 8, 5, 2)`，其中 radix-5 stage 使用 QR fallback base mixer。这避免了 padding，同时保持正交和 staged execution。

论文还提供 Kronecker fallback，用于更接近 QuIP# 的非 2 的幂 Hadamard convention，并降低 learnable parameter 数量；但 Mixed-Radix 更通用、更 expressive。

### 6. Fitting objective

直接把完整 QuIP# LDLQ solver 嵌入 rotation optimizer 过于昂贵。HARP 使用轻量 surrogate：

1. 对当前旋转后的权重 `W~(Θ)` 计算 blockwise codebook target `Q(W~(Θ))`，并 stop-gradient；
2. 定义量化误差：
   ```text
   Δ(Θ) = W~(Θ) - Q(W~(Θ))
   ```
3. 用 rotated Hessian diagonal 作为权重，得到 diagonal Hessian-weighted reconstruction proxy：
   ```text
   L_diag(Θ) = mean_{i,j} Δ_{ij}(Θ)^2 · normalized(|H~_{jj}(Θ)|)
   ```
4. 额外惩罚 off-block Hessian energy，使 curvature 更对齐 quantizer 的 contiguous block partition：
   ```text
   R_bd(Θ_V)
   ```
5. 总 fitting loss：
   ```text
   L_fit(Θ) = L_diag(Θ) + λ_bd R_bd(Θ_V)
   ```

使用 Adam 从 exact RHT initialization 优化 `Θ_U, Θ_V`。

### 7. 参数存储与部署

训练后，HARP 参数可以 int8 存储。每个 block 的 Givens angle 或 Cayley matrix 上三角 entries 用 per-block scale 量化，runtime 时重建正交 block。实验显示 int8 parameter storage 对 perplexity 影响很小，同时降低 processor overhead。

部署速度上，HARP 对每个 stride stage 使用 fused Triton GPU kernel，避免 materialized permutations；相对未融合实现延迟降低约 20%。

## 实验设置

### 比较范围

论文主要隔离一个组件：固定 QuIP#-style backend 中的 incoherence processor。因此主比较是：

- QuIP# with fixed RHT
- QuIP# with HARP

保持 codebooks、solver、random signs、calibration statistics 和 evaluation pipeline 不变。论文还测试了将 HARP 插入 QTIP backend 的 portability。

### Models

- Llama 3.2：1B、3B
- Llama 2：7B、13B、70B

### Metrics

- Perplexity：Wikitext2、C4
- Zero-shot accuracy：ARC-Challenge、ARC-Easy、PIQA、WinoGrande（lm_eval）
- Single-token latency：RTX 5080，batch size 1，sequence length 1，CUDA graphs + SDPA

### Bitwidths

- 2-bit、3-bit、4-bit
- 报告 effective BPP，包含 HARP processor parameters / metadata overhead
- HARP 同时报告 floating-point parameters 与 int8 parameter-storage variant

## 主要结果

### 1. Perplexity

在 QuIP# backend 中，把 fixed RHT 替换为 HARP 后，2–4 bit 下 perplexity 普遍改善，尤其 2-bit 最明显。

部分关键结果：

- Llama 3.2 1B，2-bit：
  - QuIP# RHT：W2 `26.27`，C4 `25.24`
  - HARP Mixed-Radix：W2 `22.30`，C4 `22.57`
  - HARP + int8 params：W2 `22.32`，C4 `22.57`
- Llama 3.2 3B，2-bit：
  - RHT：W2 `16.59`，C4 `15.88`
  - HARP：W2 `15.02`，C4 `14.77`
- Llama 2 7B，2-bit：
  - RHT：W2 `8.22`，C4 `10.86`
  - HARP：W2 `7.23`，C4 `9.49`
- Llama 2 70B，2-bit：
  - RHT：W2 `4.16`，C4 `6.01`
  - HARP：W2 `4.01`，C4 `5.81`

3-bit 与 4-bit 下也有稳定改善，但随着 RHT baseline 接近 FP16，绝对改进自然缩小。

### 2. Zero-shot accuracy

在 Llama 2 的 zero-shot evaluation 中，HARP 在 2-bit 下多数任务优于 RHT，尤其 ARC 相关任务提升明显：

- Llama 2 7B，2-bit：ARC-Challenge 从 `29.7` 提升到 `33.0`，ARC-Easy 从 `56.7` 提升到 `63.7`；
- Llama 2 13B，2-bit：ARC-Challenge 从 `33.8` 到 `36.4`，WinoGrande 从 `64.3` 到 `67.2`；
- Llama 2 70B，2-bit：ARC-Challenge 从 `47.4` 到 `48.5`，ARC-Easy 从 `76.9` 到 `77.8`。

说明 perplexity 改进不是单纯 likelihood artifact，也能转化到下游选择题任务。

### 3. Inference latency

RTX 5080 单 token latency：

- Llama 2 7B：
  - FP16：`61 tok/s`
  - QuIP# RHT 2-bit：`142 tok/s`
  - HARP 2-bit：`128 tok/s`
  - HARP 4-bit：`91 tok/s`
- Llama 2 13B：
  - FP16：OOM
  - HARP 2-bit：`84 tok/s`
  - HARP 4-bit：`60 tok/s`

HARP 相比固定 RHT 有一定额外 staged-rotation overhead，但仍大幅快于 FP16，并能让 13B 在 16GB VRAM 设置中可运行。

### 4. Backend portability

在 QTIP backend 中，HARP 也改善 perplexity：

- Llama 2 7B，2-bit：QTIP RHT W2/C4 为 `6.87/9.00`，QTIP + HARP 为 `6.62/8.79`；
- Llama 2 13B，2-bit：`5.64/7.46` 改善到 `5.51/7.34`。

这说明 HARP 不是 QuIP# 专用 trick，而是可作为 RHT-based PTQ backend 的通用 processor module。

### 5. 与 published baselines 的 context-matched 比较

在 Llama 2 context length 2048 的 2-bit weight-only PTQ 对比中，HARP 优于 QuIP#、GPTQ、OmniQuant 等：

- Llama 2 7B：HARP W2/C4 `7.85/9.68`，QuIP# `8.95/11.22`，OmniQuant `11.06/15.02`；
- Llama 2 13B：HARP `6.13/7.87`，QuIP# `6.52/8.32`。

AWQ/GPTQ 在 2-bit 下明显崩溃或退化，说明极低比特下 structured rotation + vector/backend-aware quantization 的必要性。

## 优点与局限

### 优点

1. **保持 full-precision 等价**：HARP 是 exact orthogonal change of basis，不改变未量化模型函数。
2. **drop-in 升级 RHT**：初始化等价于 RHT，适合替换 QuIP#/QTIP 这类 Hadamard-based pipeline。
3. **结构化高效**：用 butterfly-like stages 避免 dense rotation 的 `O(d^2)` 成本。
4. **data/backend-aware**：从 calibration data 学习适配层、曲率和 quantizer block structure 的旋转。
5. **实验覆盖较广**：1B–70B、2–4 bit、perplexity、zero-shot、latency、QTIP portability 均有结果。
6. **部署意识强**：考虑 int8 processor parameter storage 和 fused Triton kernels。

### 局限

1. **需要额外 calibration/fitting 成本**：相比固定 RHT，HARP 需要逐层优化 rotation 参数；虽然是一次性成本，但工程流程更复杂。
2. **主要作用于 weight-only / PTQ processor**：它不是端到端 activation/KV quantization 方案，和 OScaR 这类 KV cache quantization 的作用位置不同。
3. **依赖后端暴露 incoherence-processing step**：最适合 QuIP#/QTIP 风格 pipeline；对普通 scalar RTN/GPTQ pipeline 需要适配。
4. **latency 不如固定 RHT**：虽然仍快于 FP16，但 staged rotations 相对 RHT 有 8–15% 左右的额外 overhead。
5. **2-bit 仍有 quality gap**：HARP 显著恢复 FP16 gap，但小模型 2-bit 下与 FP16 仍有明显差距。
6. **serving 场景评估有限**：主要是 single-token microbenchmark，缺少 batch serving、prefill/decode 分离、长上下文吞吐等系统级评估。

## 与 EfficientPaper 主题的关系

HARP 属于 **quantization**，并与 hardware-aware algorithm design、structured transforms、post-training compression 相关。

它和 EfficientPaper 最近新增的 OScaR 形成有趣对照：

- OScaR 处理 KV cache quantization，发现 TNI 是 per-channel Key INT2 的瓶颈，用 Canalized Rotation + Omni-Token Scaling 改善 runtime KV 压缩；
- HARP 处理 weight PTQ 中的 fixed RHT limitation，用 learnable structured orthogonal rotations 改善 weight/codebook/backend 对齐；
- 两者都说明：极低比特量化的关键已经从“是否使用 Hadamard/rotation”推进到“rotation 如何适配具体数值结构与硬件后端”。

HARP 对量化方向的研究启示是：固定随机变换是强 baseline，但未必是终点。可学习、结构化、exact-equivalent 的 rotation processor 可以在保留部署效率的同时，把 calibration statistics 和 quantizer structure 引入低比特模型准备流程。

## 可复现/实现要点

1. 从已有 QuIP#/QTIP-style RHT pipeline 出发，替换其中 fixed Hadamard mixer。
2. 对每个 linear layer 构造 input-side `V` 和 output-side `U` processors。
3. 选择 Mixed-Radix schedule，优先 radix 8；非 2 的幂维度用 QR fallback 或 Kronecker fallback。
4. block kernel 使用 `Q(θ)G_b`，并初始化 `θ=0` 以恢复 RHT。
5. 用 calibration inputs 估计 `H = E[xx^T]`。
6. 优化 `L_fit = L_diag + λ_bd R_bd`，其中 codebook assignment stop-gradient。
7. 量化主权重后，将 HARP 参数 int8 打包，runtime 重建 block-orthogonal kernels。
8. 部署时用 fused kernels 处理 stride-stage transforms，避免显式 permutation/materialization。
9. 验证时同时报告 BPP overhead，否则 HARP 参数会让 nominal bitwidth 不公平。

## 个人备注

- HARP 的价值在于“learned but exact”：它不像非正交 data-aware transform 那样改变 full-precision 函数，也不像 SpinQuant 那样需要把 rotation 深度嵌入 Transformer graph，而是作为 PTQ backend 的可替换 processor。
- 这篇和 OScaR 都在用 Hadamard/rotation，但问题完全不同：OScaR 是 runtime KV cache INT2 量化中的 token norm imbalance；HARP 是 weight PTQ 中的 curvature/outlier/block-codebook alignment。不要把二者混为“都是旋转量化”。
- 很值得探索 HARP-style learnable rotation 能否扩展到 KV cache quantization：例如在不破坏 attention 等价性的条件下，为 Key/Value 或不同 token group 学习更适合 INT2 的 structured transform。
- 另一个潜在方向是把 HARP 的 off-block Hessian energy 思路迁移到 hardware block layout：不仅让量化误差小，还让 Tensor Core / memory layout 友好。
