# Accelerating Speculative Decoding with Block Diffusion Draft Trees

> Liran Ringel, Yaniv Romano

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

推测解码通过轻量 drafter 提前提出多个未来 token，再由目标模型一次前向传播并行验证，从而加速自回归语言模型。DFlash 表明，block diffusion drafter 可以在单次前向传播中生成整个 draft block，并在投机解码中超过 EAGLE-3 等强自回归 drafter。但 vanilla DFlash 每轮仍然只验证一条 drafted trajectory，没有充分利用 block diffusion drafter 在每个未来位置输出的完整 token 分布。

DDTree（Diffusion Draft Tree）提出直接从 block diffusion drafter 的 per-position distributions 构造 draft tree。在固定节点预算下，DDTree 用一个简单的 best-first heap 算法选择最有可能匹配目标模型的 continuation prefixes；选出的树通过 ancestor-only attention mask 在一次目标模型前向传播中高效验证。由于它建立在 DFlash 之上，DDTree 在保留单次 drafter pass 低成本的同时，显著提高接受长度和端到端加速。

## 一句话总结

DDTree 把 DFlash 从“单路径并行 draft”升级为“单次 block diffusion pass 生成多路径 draft tree”：用 per-position marginal probabilities 的乘积作为 surrogate prefix probability，在固定 tree-node budget 下选择 top-B 前缀，并用 tree attention 一次验证，从而让 DFlash 不再浪费其它高概率 continuation。

## 背景与问题

投机解码的收益取决于两件事：

1. **drafter 足够便宜**，否则生成 draft 的成本会抵消目标模型并行验证带来的收益；
2. **draft 足够准确**，目标模型才能接受更长前缀，减少每个生成 token 所需的目标模型调用次数。

DFlash 的优势在于 block diffusion drafter：它可以在一次 forward pass 中预测未来 L 个位置的 token 分布，而不是像 EAGLE-3 那样自回归逐 token 生成。这样 draft cost 基本不随 block length 线性增长，非常适合 GPU 并行执行。

但 vanilla DFlash 的使用方式仍然保守：虽然 drafter 为每个未来位置输出了完整分布 `q_i(v)`，DFlash 最终只采样/选择一条 continuation 去验证。换句话说，DFlash 的模型输出中包含很多 plausible alternatives，但每轮只拿其中一条路径给目标模型看。如果这条路径早早偏离目标模型，剩余位置的高概率备选 continuation 就被浪费。

DDTree 的核心问题是：**如何在不增加 drafter forward 次数的前提下，把 DFlash 的 per-position distributions 转化为一棵 compact draft tree，让目标模型一次验证多条候选路径？**

## 核心方法

DDTree 每轮推测解码包含四步：

1. 运行一次 block diffusion drafter，得到未来 L 个位置的 marginal distributions `q_i(·)`。
2. 在节点预算 `B` 下，从这些分布构造一棵 draft tree。
3. 将 tree flatten 成目标模型输入，并使用 ancestor-only tree attention 一次前向传播验证整棵树。
4. 按目标模型自己的解码规则沿树行走：如果目标模型选择的 token 是当前节点的 child，就接受并继续；否则停止，已匹配路径写入输出，第一个未匹配 token 成为下一轮 bonus token。

### Block diffusion drafter 输出的特殊性

给定当前 context `c` 和上一轮目标模型产生的 bonus token `b`，block diffusion drafter 接收形如 `[b, MASK, ..., MASK]` 的 block，并并行预测未来 L 个位置的 logits：

```text
ℓ_i ∈ R^{|V|},  i = 1,...,L
q_i(v) = softmax(ℓ_i)_v
```

关键点是：这些 `q_i` 是 **position-wise marginals**，不是路径条件分布。即位置 i 的预测没有显式 conditioning on 前面已经选择的 draft token `y_1,...,y_{i-1}`。因此，目标模型真实分布是 autoregressive 的：

```text
p(y_1:L | c,b) = ∏ p(y_i | c,b,y_1:i-1)
```

而 block diffusion drafter 可提供的自然 surrogate 是 factorized distribution：

```text
Q(y_1:L | c,b) = ∏ q_i(y_i | c,b)
```

DDTree 接受这个限制：它不试图恢复不可得的 target conditional path probabilities，而是在 `Q` 下构造最优 tree。

## 技术细节

### Surrogate objective

把 draft tree `T` 视为一组 prefix-closed candidate prefixes。深度 d 的节点表示一个 continuation prefix：

```text
u = (u_1, ..., u_d)
```

对任意完整 continuation `y_1:L`，定义 tree acceptance length：

```text
α_T(y_1:L) = max { d : y_1:d ∈ T }
```

理想目标是最大化目标模型分布 `p` 下的期望接受长度，但 `p` 的 path-conditioned continuation probabilities 在构树时不可得。因此 DDTree 使用 drafter factorized distribution `Q` 作为 surrogate：

```text
max_T E_{Y~Q}[α_T(Y)]
subject to |T| ≤ B, T prefix-closed
```

论文证明该目标可以分解为 tree 中所有节点 prefix mass 的加和：

```text
E_{Y~Q}[α_T(Y)] = Σ_{u∈T} q(u)
q(u) = ∏_{i=1}^{|u|} q_i(u_i)
```

因此，在 surrogate 下，最优 tree 就是在满足 prefix-closure 和节点预算的前提下，选择 prefix probability 最大的 B 个 prefixes。

### Best-first heap algorithm

直接枚举所有 prefixes 不可行，因为深度 L 的 prefix 空间大小指数增长。DDTree 利用排序后的 per-position top tokens 和 max-heap 做 best-first enumeration。

对每个位置 i，将 token 按 `q_i` 从大到小排序：

```text
v_i^(1), v_i^(2), ...
```

一个 prefix 可以用 rank tuple 表示：

```text
ρ = (ρ_1, ..., ρ_d)
```

对应 prefix token 为 `(v_1^(ρ_1), ..., v_d^(ρ_d))`，log probability 为：

```text
σ(ρ) = Σ log q_i^(ρ_i)
```

算法从 `(1)` 开始，每次 pop 当前 log probability 最大的 tuple，加入 tree，然后最多 push 两类候选：

- **next sibling**：把最后一位 rank 从 `ρ_d` 改为 `ρ_d + 1`，探索当前位置的下一个 token；
- **first child**：追加下一深度的 rank-1 token，扩展当前 prefix。

重复直到得到 B 个节点。论文证明该 best-first heap 算法返回 surrogate objective 下的 optimal valid draft tree。复杂度为 `O(B log B)`，因为最多 B 次 pop、2B 次 push，heap size 也是 `O(B)`。

### Efficient verification and KV cache update

构好 draft tree 后，DDTree 将其 flatten 成目标模型输入序列。每个节点的位置 id 由 tree depth 决定，以保证 positional encoding 正确。attention mask 使用 tree attention：每个 draft node 可以 attend 到：

- 过去 context 的 KV cache；
- root / bonus token；
- 自己的 ancestors；
- 自己。

它不能 attend 到 sibling 或 unrelated branches。这等价于 ancestor-only attention mask，可以让目标模型在一次 forward pass 中同时验证所有分支。

验证后，目标模型按自己的 greedy 或 sampling rule 在 tree 中 walk：若目标 token 匹配当前节点的 child，则接受该 child 并继续；若没有 child 匹配，则停止。最后只保留 accepted path 的 KV cache，丢弃未被接受分支的 cache，并把第一个 unmatched target token 作为下一轮 bonus token。

## 实验设置

### 模型

论文评估三个 target models，每个都配套对应 DFlash checkpoint：

- Qwen3-4B
- Qwen3-8B
- Qwen3-Coder-30B-A3B-Instruct

DFlash checkpoint 来自：`https://huggingface.co/collections/z-lab/dflash`。

### 任务

benchmark 覆盖 reasoning、code、general instruction / dialogue：

- reasoning：MATH-500、GSM8K、AIME 2024、AIME 2025
- code：HumanEval、MBPP、LiveCodeBench、SWE-bench Lite
- general / dialogue：MT-Bench、Alpaca

### 硬件与指标

- 硬件：8 × NVIDIA H200 GPUs
- 温度：`temperature=0.0` 和 `temperature=1.0`
- 指标：相对 autoregressive decoding 的 speedup、平均 acceptance length `τ`（包含 bonus token）、acceptance length histogram
- DDTree node budget 从 `{16, 32, 64, 128, 256, 512, 1024}` 中选择；主结果中每个 dataset-model-temperature 使用最佳 budget。

## 主要结果

### 全面优于 vanilla DFlash

Table 1 覆盖 `10 datasets × 3 models × 2 temperatures = 60` 个设置。论文报告 DDTree 在全部 60 个设置中都提升了 DFlash。

### Temperature = 0.0 代表结果

在 greedy decoding 下，DDTree 对 DFlash 的提升非常稳定：

- **Qwen3-4B**：
  - MATH-500：DFlash `5.54× / τ=7.72` → DDTree `7.50× / τ=10.71`
  - GSM8K：`4.77× / τ=6.51` → `6.58× / τ=9.33`
  - HumanEval：`4.81× / τ=6.62` → `6.81× / τ=9.44`
  - AIME 2025：`5.33× / τ=7.37` → `7.23× / τ=10.23`

- **Qwen3-8B**：
  - MATH-500：`5.56× / τ=7.79` → `7.52× / τ=10.73`
  - GSM8K：`4.78× / τ=6.57` → `6.75× / τ=9.54`
  - HumanEval：`4.84× / τ=6.61` → `6.90× / τ=9.67`
  - LiveCodeBench：`5.02× / τ=7.22` → `7.10× / τ=10.28`

- **Qwen3-Coder-30B-A3B-Instruct**：
  - HumanEval：`6.09× / τ=8.02` → `8.22× / τ=10.72`
  - MBPP：`5.61× / τ=7.19` → `7.68× / τ=9.94`
  - MATH-500：`4.29× / τ=5.58` → `6.21× / τ=8.10`

Figure 1 中最高 speedup 约为 8.2×（HumanEval, Qwen3-Coder-30B-A3B-Instruct）。

### Temperature = 1.0 下仍然有效

sampling decoding 更难，因为目标模型行为更随机，但 DDTree 仍在全部设置中提升 DFlash。例如：

- Qwen3-8B GSM8K：DFlash `4.33× / τ=5.93` → DDTree `6.27× / τ=8.95`
- Qwen3-4B MATH-500：`4.65× / τ=6.60` → `6.60× / τ=9.61`
- Qwen3-Coder HumanEval：`5.64× / τ=7.60` → `7.88× / τ=10.42`

### Budget-quality tradeoff

在 Qwen3-8B + MATH-500 + temperature 0.0 case study 中，随着 DDTree node budget 增大，acceptance length 持续上升，但 end-to-end speedup 在中等 budget（约 256–512）附近达到峰值。继续把 budget 增到 1024 会提高接受长度，但 verifier 要处理更多 tree nodes，目标模型 forward 成本增加，最终 speedup 反而不划算。

这说明 DDTree 的关键不是“树越大越好”，而是构造一个 **front-heavy tree**：优先覆盖最可能被目标模型沿着走下去的高概率 prefixes，不把预算浪费在低概率分支上。

### Acceptance length distribution

在 Qwen3-8B + MATH-500 + temperature 0.0 下，DDTree（B=512）显著把 acceptance length histogram 的质量推向长前缀：短于 4 的接受长度更少，完整 block acceptance（长度 16）更常见。这解释了端到端加速来源：更长接受前缀意味着目标模型每生成一个 token 所需的 verification rounds 更少。

## 优点与局限

### 优点

1. **充分利用 DFlash 输出**：不再只取一条 trajectory，而是把每个位置的分布用于构造多路径 tree。
2. **不增加 drafter forward 次数**：所有候选分支仍来自一次 block diffusion pass，保持 DFlash 的低 draft latency。
3. **目标清晰且有理论保证**：在 factorized drafter distribution `Q` 下，top-B prefix tree 是 surrogate expected acceptance length 的最优解。
4. **算法简单**：best-first heap 构树复杂度 `O(B log B)`，实现上比需要外部 N-gram trie / continuity score 的方法更直接。
5. **验证兼容 tree attention**：一次 target forward 验证整棵树，避免逐分支调用目标模型。
6. **实证提升全面**：60 个 dataset-model-temperature 设置全部优于 DFlash。

### 局限

1. **surrogate 与真实目标模型分布不一致**：DDTree 优化的是 drafter factorized distribution `Q` 下的期望接受长度，而不是真实 autoregressive target distribution `p`。如果 DFlash marginals 与 target conditional path distribution 偏差很大，top-B prefixes 未必是 target-optimal。
2. **marginal independence 假设较弱**：block diffusion 输出的 per-position marginals 不条件化于前面选择的 draft token，因此路径概率乘积可能高估某些不连贯组合。
3. **节点预算依赖硬件与实现**：最佳 budget 在 256–512 左右只是论文实验设置下的结果；不同 GPU、batch size、kernel 实现和模型大小会改变 verifier overhead 与 acceptance gain 的 tradeoff。
4. **额外 tree attention / cache compaction 工程复杂度**：虽然理论上一轮 target forward 即可验证，但实际系统需要高效 flatten tree、构造 mask、管理 position ids，并在验证后只保留 accepted path 的 KV cache。
5. **建立在 DFlash checkpoint 之上**：DDTree 不是独立 drafter，而是 DFlash 的 tree-verification 扩展；如果没有高质量 block diffusion drafter，收益有限。

## 与 EfficientPaper 主题的关系

DDTree 属于 **speculative_decoding**，是 DFlash 与 Domino 之后同一条线的继续推进。

如果说：

- **DFlash** 解决的是 `T_draft`：用 block diffusion 一次 forward 生成整个 draft block，避免自回归 draft 的串行成本；
- **Domino** 解决的是 draft quality：给并行 base logits 加轻量 causal residual，恢复部分因果依赖；
- **DDTree** 解决的是 verification utilization：同样一次 block diffusion pass，不只验证一条 path，而是用 tree attention 验证多个高概率 continuations。

那么方向 8 的研究脉络已经很清晰：投机解码正在从“单一自回归 drafter”转向 **one-pass parallel drafting + structured multi-path verification + lightweight correction**。下一步的核心问题是如何联合优化：

```text
draft cost + tree verification cost + causal correction cost + acceptance-length gain
```

## 可复现/实现要点

1. 需要一个 DFlash-style block diffusion drafter，输出每个未来位置的 logits / probabilities。
2. 对每个 position 取 top-K tokens，其中 `K = min(B, |V|)`。
3. 用 rank tuple 表示 prefix，用 log probability sum 做 heap score。
4. best-first pop B 次，生成 top-B prefixes；每次 pop 后 push sibling 和 first child。
5. 构造 prefix-closed tree，并 flatten 成 target-model input。
6. position ids 按 tree depth 设置，而不是 flatten 后的线性位置。
7. attention mask 必须是 ancestor-only：节点只能看 context、root、ancestors 和自己。
8. verification walk 后只保留 accepted path 的 KV cache，未接受分支的 KV 需要丢弃或 compact。
9. node budget 需要按硬件与任务调参；过小不能覆盖足够 alternatives，过大则 verifier overhead 吃掉收益。

## 个人备注

- DDTree 的重要性在于它指出 DFlash 的瓶颈已经不是 draft generation，而是 **如何消费 drafter 输出的信息**。当一次 diffusion pass 给出的是整段 marginals，单路径采样显然信息利用率不足。
- DDTree 与 Domino 是互补的：Domino 让每个 position 的分布更接近 target conditional behavior；DDTree 则从这些分布中构建多路径 tree。二者结合可能进一步提高 acceptance length，但也会提高 tree/correction/runtime 复杂度。
- 论文使用 factorized `Q` 构树，这简洁且可证明最优，但也留下了研究空间：能否引入轻量 pairwise / causal consistency score，在不重新自回归 forward 的情况下减少 marginal product 带来的不连贯路径？
- 从系统角度，DDTree 把投机解码的核心优化对象从“draft token 数”变成“tree node budget allocation”。这与 KV cache 临时分支、tree attention mask、verification batch shape 和 GPU kernel efficiency 强相关，值得和 SGLang / vLLM 的 serving runtime 共同设计。
