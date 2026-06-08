# PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU

![](../../blank.jpg)

## 一句话总结

PowerInfer 利用 LLM 推理中神经元激活的幂律分布特性，设计了一种 GPU-CPU 混合推理引擎，将高频激活的"热"神经元预加载到 GPU 上，低频激活的"冷"神经元由 CPU 计算，从而在消费级 GPU（如 RTX 4090）上实现高达 11.69× 的推理加速，性能仅比服务器级 A100 GPU 低 18%。

## 摘要翻译

本文介绍了 PowerInfer，一种在配备单个消费级 GPU 的个人电脑（PC）上运行的高速大语言模型（LLM）推理引擎。PowerInfer 设计的核心在于利用 LLM 推理中固有的高局部性，其特征是神经元激活的幂律分布。这种分布表明，一小部分被称为"热神经元"的神经元在不同输入间持续被激活，而大多数"冷神经元"的激活则取决于具体输入。PowerInfer 利用这一洞察设计了一种 GPU-CPU 混合推理引擎：热激活神经元被预加载到 GPU 上以便快速访问，而冷激活神经元则在 CPU 上计算，从而显著减少 GPU 内存需求和 CPU-GPU 数据传输。PowerInfer 进一步集成了自适应预测器和神经元感知稀疏算子，优化了神经元激活和计算稀疏性的效率。评估表明，PowerInfer 在单个 NVIDIA RTX 4090 GPU 上，跨多种 LLM（包括 OPT-175B），平均 token 生成速率达到 13.20 tokens/s，峰值达 29.08 tokens/s，仅比顶级服务器级 A100 GPU 低 18%。这一性能显著优于 llama.cpp，最高提升达 11.69 倍，同时保持了模型精度。

## 研究动机

在消费级 GPU 上部署大语言模型面临重大挑战：

1. **内存瓶颈**：LLM 作为自回归 Transformer，每次生成 token 都需要访问整个模型（数百亿参数），推理过程根本受限于 GPU 内存容量。例如，一个 OPT-66B 模型在 4 位精度下仍需约 40GB 内存，超过 NVIDIA RTX 4090 的 24GB 容量。

2. **现有方案不足**：
   - **模型压缩**（量化、蒸馏、剪枝）即使深度压缩后模型仍然太大，无法放入消费级 GPU。
   - **GPU-CPU 卸载**（如 llama.cpp）按 Transformer 层级别分配模型参数，但受慢速 PCIe 互联和 CPU 有限计算能力的限制，导致高推理延迟。对于 30B 参数模型，llama.cpp 仅将 37% 的模型放在 GPU 上，98% 的计算任务由 CPU 承担。
   - **FlexGen** 等 GPU-Centric 方案在 batch size=1 时超过 99.5% 的时间花在 CPU-GPU 数据传输上。
   - **DejaVu** 虽然利用激活稀疏性加速推理，但在消费级 GPU 上因频繁的 CPU-GPU 数据传输而表现不佳。

3. **关键洞察——局部性不匹配**：当前硬件架构针对数据局部性优化，但 LLM 推理中需要访问的参数量过大，导致 GPU-CPU 内存层级结构无法高效利用局部性。

4. **LLM 推理固有的高局部性**：
   - **幂律激活分布**：在 OPT-30B 中，26% 的神经元贡献了 80% 的激活；在 LLaMA(ReGLU)-70B 中，43% 的神经元贡献了 80% 的激活。
   - **快速 CPU 计算**：当激活神经元数量较少（如 10%）且 batch size 较小时，直接在 CPU 上计算比传输到 GPU 再计算更快（使用 AVX2 向量扩展）。

## 方法（技术细节）

### 1. 整体架构

PowerInfer 由离线（Offline）和在线（Online）两部分组成：

**离线组件（LLM Profiler & Policy Solver）**：
- **LLM Profiler**：使用通用数据集（如 C4）生成请求，监控所有层的神经元激活，统计每个神经元的激活频率。
- **Policy Solver**：基于整数线性规划（ILP）将神经元分为"热"和"冷"两类，将高频激活神经元分配给 GPU，低频的分配给 CPU。目标函数最大化 GPU 上神经元的总影响力，同时考虑通信约束和内存约束。
- ILP 求解时将 64 个相似影响力的神经元聚合为一个 batch，将总神经元数从数百万降低到约数万，使 ILP 求解时间约 10 秒。

**在线组件（Neuron-aware LLM Inference Engine）**：
- 根据离线求解器的输出将神经元分配到 GPU 和 CPU。
- 创建 GPU 和 CPU executor（pthread 线程），管理并发的 CPU-GPU 计算。
- 构建计算有向无环图（DAG），每个节点代表一个推理算子，存储在 CPU 内存的全局队列中。
- GPU executor 使用 cudaLaunchKernel 等 API 启动 GPU 算子，CPU executor 协调空闲 CPU 核心进行计算。

### 2. 自适应稀疏预测器（Adaptive Sparsity Predictors）

- PowerInfer 使用 MLP 预测器预测哪些神经元将被激活（参考 DejaVu），但采用自适应训练方法。
- **核心洞察**：预测器大小受层稀疏度和偏斜度（skewness）影响。高稀疏度层可使用较小的预测器，高偏斜度层也可使用较小预测器。
- **自适应迭代训练**：基于层稀疏度确定基线模型大小，然后根据偏斜度迭代调整隐藏层维度。高偏斜度层逐步缩小隐藏层大小（直到精度低于 95%），低偏斜度层增大维度。
- 最终将预测器参数限制在 LLM 总参数的 10% 以内（对比 OPT-175B 原始预测器需约 27GB GPU 内存）。

### 3. 神经元放置与管理（Neuron Placement & Management）

- 创建两个神经元表（CPU 和 GPU 各一个），映射每个神经元到其在权重矩阵中的原始位置。
- 每个神经元表的额外内存开销极小（OPT-175B 模型约 9MB，而模型存储为 350GB）。
- ILP 模型考虑因素：
  - **通信约束**：每个层至少需要分配 Cl 个神经元到 GPU，以补偿同步开销。
  - **内存约束**：分配到各处理单元的神经元不能超过其内存容量。
  - **ILP 优化**：使用整数线性规划最大化 GPU 上神经元的总影响力。

### 4. GPU-CPU 混合执行（GPU-CPU Hybrid Execution）

- GPU 和 CPU 独立计算各自分配的激活神经元，然后在 GPU 上合并结果。
- **选择性同步策略**：当 CPU executor 没有激活神经元时，跳过结果同步，允许其继续处理后续 block。
- **DAG 调度**：构建计算 DAG，每个算子标记前置依赖，CPU 和 GPU executor 从全局队列拉取算子，检查依赖后分配到适当的处理单元。

### 5. 神经元感知算子（Neuron-aware Operators）

**传统稀疏算子的问题**：
- cuSPARSE 等通用库设计需要跟踪每个非零元素并转换矩阵格式，存在显著性能开销。
- PIT 等 JIT 编译器仅针对 GPU 优化，不支持 CPU-GPU 混合执行。

**PowerInfer 神经元感知算子**：
- 直接操作矩阵中的单个行/列（神经元），无需运行时转换为稠密格式。
- **GPU 算子**：在 batch size 较小时，基于向量-向量计算比矩阵-向量计算更高效。所有线程块可并发检查神经元激活状态并计算对应的向量。
- **CPU 算子**：将神经元分配给多个核心，每个核心仅处理激活的神经元，利用 AVX2 向量扩展优化向量-向量计算。
- 实现了 10 个神经元感知算子，CPU 上在稀疏度低于 10% 时即优于稠密矩阵乘法。

### 6. 实现细节

- 在 llama.cpp 基础上扩展 4,200 行 C++/CUDA 代码。
- 在 transformers 框架上增加约 400 行 Python 代码作为离线 profiler 和 solver。
- 支持的 LLM 家族：OPT（7B-175B）、LLaMA（7B-70B）、Falcon-40B。
- 支持的消费级 GPU：NVIDIA RTX 4090 和 RTX 2080Ti。
- KV cache 保持在 CPU 内存中，为热激活神经元腾出更多 GPU 内存。

## 实验结果

### 实验设置

- **PC-High**：Intel i9-13900K（8 核 5.4GHz）+ 192GB 内存 + NVIDIA RTX 4090（24GB，1TB/s 带宽，PCIe 4.0）
- **PC-Low**：Intel i7-12700K（8 核 4.9GHz）+ 64GB 内存 + NVIDIA RTX 2080Ti（11GB，616GB/s 带宽，PCIe 3.0）
- **模型**：OPT（6.7B-175B）、Falcon(ReLU)-40B、LLaMA(ReGLU)-70B，使用 FP16 和 INT4 量化
- **数据集**：ChatGPT prompts、Alpaca
- **基线**：llama.cpp

### 端到端性能（FP16）

- **PC-High**：平均 8.32 tokens/s，最高 16.06 tokens/s
- **对比 llama.cpp**：平均加速 7.23×，最高 11.69×（Falcon-40B）
- **PC-Low**：平均加速 5.01×，峰值 7.06×
- **GPU 负载提升**：从 llama.cpp 的平均 20% 提升到 PowerInfer 的 70%（PC-High）

### 量化模型（INT4）性能

- **PC-High**：平均 13.20 tokens/s，峰值 29.08 tokens/s
- **对比 llama.cpp**：平均加速 2.89×，最高 4.28×
- **PC-Low**：平均加速 5.01×，峰值 8.00×
- **OPT-175B**：在 PC-High 上接近 2 tokens/s，超越 llama.cpp 2.66×

### 与 A100 对比

- PowerInfer（RTX 4090）vs vLLM（A100）：仅慢 18%-29%（输入长度 1-64）
- llama.cpp（RTX 4090）vs vLLM（A100）：慢 92%-93%

### 批处理推理

- batch size=1 时加速 10.73×
- batch size=32 时仍保持 4.38× 加速

### 消融实验

- **+PO**（预测器 + 神经元感知算子）：OPT-30B 加速 1.98×，OPT-66B 加速 2.00×
- **+Engine**（混合执行引擎）：OPT-30B 加速 9.97×，OPT-66B 加速 3.43×
- **+Policy**（优化的放置策略）：OPT-30B 加速 10.47×，OPT-66B 加速 3.67×

### 神经元感知算子性能

- CPU：在稀疏度低于 10% 时即优于稠密矩阵乘法（传统稀疏算子需稀疏度超过 87% 才优于稠密计算）
- GPU：与 PIT 性能相当，但具有统一的 CPU-GPU 框架

### 预测器开销

- 预测器执行时间占总推理时间不到 10%

### 模型精度

- PowerInfer 对模型精度影响可忽略不计（即使选择性跳过预测不活跃的神经元）
- 各下游任务（PIQA、Winogrande、RTE、COPA）精度波动极小

## 优势

1. **显著的推理加速**：相比 llama.cpp 最高提升 11.69×，在消费级 GPU 上接近服务器级 A100 性能（仅差 18%）。
2. **低内存需求**：利用神经元激活的幂律分布，将热神经元预加载到 GPU，大幅减少 GPU 内存需求。
3. **无需额外压缩**：基于 LLM 推理的自然稀疏性，保持模型原始精度，无需牺牲精度。
4. **自适应预测器**：预测器参数仅占 LLM 总参数的 10%，在保持 95% 预测精度的同时大幅降低 GPU 内存开销。
5. **高效的神经元感知算子**：CPU 上在稀疏度低于 10% 时即优于稠密矩阵乘法，避免了传统稀疏算子的格式转换开销。
6. **广泛的模型支持**：支持 OPT（7B-175B）、LLaMA（7B-70B）、Falcon-40B 等主流 LLM 家族。
7. **代码开放**：源码公开，基于 llama.cpp 扩展，易于集成和部署。
8. **对量化模型的友好支持**：INT4 量化模型在 PC-High 上平均达到 13.20 tokens/s，峰值 29.08 tokens/s。
9. **批处理优化**：即使在 batch size=32 时仍保持 4.38× 加速。

## 局限

1. **长输入短输出场景**：在长输入提示、短输出长度的场景中，由于 prompt 阶段处理大量 token，每个 token 激活独特的神经元集合，显著降低激活稀疏性，性能提升有限。
2. **CPU 成为瓶颈**：当模型内存需求远超 GPU 容量时（如在 11GB 2080Ti 上运行 60GB 模型），GPU 能分配的热激活神经元减少，CPU 承担更多计算，成为性能瓶颈。
3. **低端 PC 性能受限**：PC-Low（11GB GPU）相比 PC-High 加速幅度较小，尤其对于 30B+ 参数的模型。
4. **离线分析开销**：需要使用通用数据集进行离线 profiling，且 ILP 求解需要额外时间（约 10 秒），虽为一次性任务但增加了部署复杂度。
5. **批处理加速随 batch size 增大而降低**：随着 batch size 增加，模型联合激活的稀疏性降低，PowerInfer 的加速比下降。
6. **预测器精度与模型精度的权衡**：虽然 PowerInfer 保持了模型精度，但预测器偶尔会遗漏部分活跃神经元，导致精度在某些任务上略有波动。
7. **不支持所有激活函数**：当前实现主要针对 ReLU 和 ReGLU 等激活函数的稀疏性，对其他激活函数的适用性有待验证。

## 与 EfficientPaper 相关的研究方向

1. **激活稀疏性（Activation Sparsity）**：PowerInfer 的核心洞察基于 LLM 推理中神经元激活的幂律分布，这与 DejaVu、PIT、brainstorm 等工作密切相关。关键词 `activation_sparsity` 是本文的核心研究方向。

2. **稀疏剪枝（Sparse Pruning）**：PowerInfer 的神经元感知算子与稀疏矩阵计算密切相关，与 SparseGPT、Wanda、SparTA、Flash-LLM 等稀疏剪枝工作形成互补关系。关键词 `sparse_pruning` 指向这一方向。

3. **GPU-CPU 混合推理**：PowerInfer 的 GPU-CPU 混合执行模型与 llama.cpp、FlexGen 等卸载策略有关，但通过神经元级别的细粒度分配实现了更优的性能。

4. **自适应稀疏预测器**：PowerInfer 的自适应预测器设计可用于优化 LLM 推理中的动态稀疏性，与 DejaVu 的固定大小预测器形成对比。

5. **神经元级稀疏算子**：PowerInfer 的神经元感知算子为稀疏矩阵乘法提供了新的实现思路，与传统稀疏算子（cuSPARSE、SparTA、Flash-LLM）形成对比。

6. **本地 LLM 推理优化**：PowerInfer 专注于在消费级硬件上运行大模型，与 vLLM、Orca 等服务端优化系统形成互补。

7. **推测推理（Speculative Inference）**：PowerInfer 未涉及推测推理，但结合推测解码可进一步提升推理速度，是潜在的研究方向。

## AI 生成声明

> **声明**：本笔记由 AI Agent（Hermes Agent）基于论文全文自动提取和生成。笔记内容来源于 arXiv 论文 PowerInfer（arXiv:2312.12456v1）的文本提取，使用 PyMuPDF (fitz) 进行 PDF 文本解析。AI 生成的翻译和总结可能存在不准确之处，请以原文为准。本笔记仅用于学术研究和学习参考，不构成任何商业建议。
