# Nested Learning: The Illusion of Deep Learning Architectures

> Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

论文提出 Nested Learning（NL）范式：将机器学习模型及其训练过程表示为一组嵌套、多层、并行的优化问题，每一层拥有自己的 context flow、目标函数、参数和更新频率。NL 认为深度学习中的架构、优化器、记忆和 in-context learning 可以放在同一套视角下理解，并以 Expressive Optimizers、Self-Modifying Learning Module、Continuum Memory System 三类设计为例，构造持续学习模块 Hope，在语言建模、知识注入、少样本泛化、持续学习和长上下文推理任务上取得有竞争力的结果。

## 一句话总结

NL 把“堆叠更多静态层”扩展为“堆叠具有独立时间尺度和 context flow 的学习层”，用多级记忆与自修改更新机制提升模型的在线适应和持续学习能力。

## 创新点

1. **统一的 Nested Learning 表示**：每个组件都可视为带内部梯度流的学习模块；架构产生 token context，优化器接收 gradient context，二者是不同层级但相互连接的优化过程。该视角把 meta-learning、in-context learning、RNN、hypernetwork 和 learned optimizer 纳入同一框架。
2. **优化器作为关联记忆**：论文将反向传播解释为把输入映射到局部 prediction error 的压缩过程，将 momentum、Adam、AdaGrad 等解释为压缩历史梯度的关联记忆。基于此提出更具表达力的更新规则，包括 Delta Gradient Descent（DGD）和多动量项的 Multi-scale Momentum Muon（M3）。
3. **多时间尺度的 Continuum Memory System（CMS）**：不再把记忆简单二分为 short-term / long-term，而是使用不同更新频率的互联记忆层；高频层负责快速适应，低频层保存更持久的知识，并允许部分恢复已被遗忘的信息。
4. **Self-Modifying Learning Module 与 Hope**：让序列模块学习自己的更新算法，并与 CMS 组合成 Hope，将多级 in-context learning、持续适应和参数内知识迁移结合起来。

## 带来什么提升

1. 在 class-incremental learning、连续学习新语言和新语料问答等任务中，Hope 整体优于普通 ICL、Elastic Weight Consolidation（EWC）和外部学习器 InCA；多级记忆显著减轻了连续学习中的性能退化。
2. 在 NIAH 长上下文实验中，Hope 在 4K/8K/16K 多种设置下保持较强检索能力。例如 UUID 单 needle 在 16K 上 Hope 为 24.8，优于 Transformer 的 21.5；multi-key、multi-query 和 multi-value 任务中，Hope 也优于其对比的现代 recurrent/deep-memory 模型。
3. 记忆层数增加通常能改善 in-context learning 和长程记忆；实验同时显示，最低频率层越慢，持久记忆更强但适应性和计算效率可能下降，最低频率为 2K 在效果与 forward 成本之间提供了较好的折中。
4. NL 给优化器设计增加了“内部计算深度”和“记忆表达力”两个轴：优化器不必只是一次梯度变换，还可以通过内部学习过程压缩更丰富的历史梯度或全局损失景观信息。

## 备注

- 这篇论文更像一个统一框架和研究路线图，而不是单一、已完全定型的模型；Hope 的实验结果依赖较高的额外记忆与更新成本，和 Transformer、外部记忆方法之间的计算/内存开销并非完全同构。
- 论文明确指出高频记忆带来快速适应，低频记忆带来持久性；实际系统需要联合优化更新频率、memory level 数量、状态大小、带宽和质量损失，而不能只增加层级。
- 对 EfficientPaper 的关联：NL 为 linear attention state、test-time learning、memory-augmented serving 和持续学习提供了共同的算法语言，值得作为“训练-推理协同”和“可学习记忆系统”的基础理论方向跟踪。
