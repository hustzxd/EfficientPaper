# SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs

> Xinrui Zheng, Dongliang Wei, Jianxiang Gao, Yixin Song, Zeyu Mi, Haibo Chen；Shanghai Jiao Tong University；FAST 2026。

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

面向内存受限的 AI PC，SolidAttention 将动态稀疏注意力与 SSD 存储管理共同设计：完整 KV 保留在 SSD，只加载当前需要的块；通过扩大传输粒度、利用选择结果的时间局部性预取，以及细粒度计算/I/O 重叠，降低单用户长上下文生成的阻塞延迟。

## 一句话总结

让稀疏注意力的访问粒度和执行时机适配 SSD，使 batch=1 的长上下文推理也能有效隐藏 KV 换入延迟。

## 创新点

1. **KV consolidator：扩大字节粒度而不扩大 token 块。** 按 token 交错存储 K/V，将原本独立的读取合并，传输单位翻倍、I/O 次数减半；预先合并 K/V 投影权重，一次矩阵乘法直接生成交错结果，注意力 kernel 通过 stride 读取，避免运行时重排。区别于单纯增大 token 块，这不会让块代表向量概括更多 token 而额外损害选择精度。
2. **Speculative prefetcher：同层上一轮预测下一轮。** 相邻 decode 迭代的块选择平均约 81% 相似。初始块与最近上下文块确定性预取，动态块沿用该层上一轮选择；正式选择后仅补读缺失块，直接覆盖误取块。KV 对齐即可，无需按原 token 顺序重排整个缓存。
3. **SSD-aware scheduler：按数据依赖组织微任务。** 将 Q/KV 投影、选择、预取、补读、注意力及写回建模成 DAG，优先关键路径与最晚启动时间更早的任务；补读与 KV 投影重叠，下一层预取与当前层计算重叠，并合并非关键写回与必要同步点，避免共享小缓冲区的数据冲突。

## 带来什么提升

1. **长上下文吞吐：** 128k 输入、batch=1 的 CUDA 实验中，相比 Offload+Sparse（SSD 卸载 + InfLLM），Llama-3.2-3B、Llama-3.1-8B、Qwen2.5-7B 分别加速 2.8、3.1、2.4 倍；SYCL 对应最高 2.1、2.5、1.9 倍。16k 下相对扩展版 FlexGen 最高 58.9 倍，但 FlexGen 在更长输入下 OOM，此数字不能替代 128k 对比。
2. **KV 驻留内存：** 相比完整 KV 驻留的 llama.cpp，三个模型最高约减少 62 倍（约 98%）。这是 KV 内存而非模型总内存；完整历史 KV 仍存在 SSD。SYCL 消融中，相对内存驻留方案的吞吐下降不超过 11%。
3. **机制收益：** 预取将 KV 加载阻塞延迟最高缩短 3.9 倍（CUDA）/3.1 倍（SYCL）；交错布局降低 attention 延迟最高 22%；细粒度重叠带来最高 25% 性能提升，同步点复用进一步降低 attention 延迟最高 22%。这些来自不同消融，不能相加。
4. **准确率：** 表 1 的综合均分接近原版，但并非所有任务无损。LongBench 分数分别为 40.10→38.25、46.75→45.35、45.77→43.75；动态稀疏选择仍属于近似注意力。

## 备注

- 实现基于 llama.cpp、liburing，提供 CUDA/SYCL 后端；权重 INT4、KV FP16，默认块大小 32。输入不短于 4k 时 attention budget 为 1k token，一半用于初始/局部块，一半用于动态选择；该 budget 不等于整个历史上下文容量。
- 两台实验笔记本实际均有 64 GB DRAM，端到端性能实验将 DRAM 使用限制为 16 GB；独显为 RTX 4070 Laptop 8 GB，SSD 为 Samsung 990 PRO。结论主要适用于低并发本地 decode，不能直接外推到高并发 serving。
- SSD 争用仍是边界：4 GB/s 后台流量使吞吐下降 58%，800k IOPS 干扰使吞吐下降 54%。扩大读取 budget 后，I/O 也可能超出计算可隐藏的窗口。
- [官方论文页](https://www.usenix.org/conference/fast26/presentation/zheng) · [PDF](https://www.usenix.org/system/files/fast26-zheng.pdf)。正文未列出官方代码仓库，元数据保留空链接。
