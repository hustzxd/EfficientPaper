# Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning

> Shan Yu, Yifan Qiao, Mingyuan Ma, Yangmin Li, Shuo Yang, Xinyuan Tong, Yang Wang, Zhiqiang Xie, Yuwei An, Shiyi Cao, Ke Bao, Deepak Vij, Xiaoning Ding, Yichen Wang, Qingda Lu, Zhong Wang, Gao Gao, Harry Xu, Junyi Shu, Jiarong Xing, Ying Sheng

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

多模型在线服务中的活跃模型集合和请求率持续变化：纯空间共享会让空闲模型占住权重/KV 显存，纯时间共享则在突发时频繁换权重并造成 SLO 违约。Prism 把 GPU 显存视为跨模型弹性资源，以 CUDA VMM memory ballooning 在不改变 attention kernel 的前提下动态伸缩模型权重与 KV Cache，再用全局模型放置和 GPU 本地请求仲裁统一空间/时间共享。

## 一句话总结

Prism 用 `kvcached` 把每个模型看到的连续虚拟显存与实际物理 HBM 解耦，使权重和 KV Cache 能在模型间按需“充气/放气”，再以 KV pressure 和请求 slack 驱动放置与调度。

## 创新点

1. **跨模型 GPU memory ballooning。** 推理引擎预留大块虚拟地址空间，物理页按需映射；`kvcached` 以 2 MB 粒度统一管理权重和异构 KV Cache，毫秒级重分配，保持 CUDA Graph/PagedAttention 兼容，集成 SGLang 只需修改 22 行。
2. **以 KV Pressure Ratio 驱动全局放置。** KVPR 用 SLO 加权的 token 显存增长速率衡量共享 GPU 压力，把高/低需求模型互补放置，并在收益足够时迁移、空闲时驱逐，避免固定空间共享和频繁 swap 的两类极端。
3. **基于 slack 的 GPU 本地仲裁。** 共享队列根据 TTFT deadline 与预计执行成本选择能按期完成的请求集合；模型激活复用预初始化 engine pool，迁移时源实例继续服务，把大部分权重/KV 搬运移出 TTFT 关键路径。

## 带来什么提升

1. 在 4 节点、32×H100、Hyperbolic/Arena-Chat 真实轨迹上，相同 GPU 数量时 TTFT SLO attainment 最高为基线 **3.3×**、TPOT 为 **2×**；保持 99% SLO 时，比 MuxServe++/静态划分多处理 **2.3×/3.5×** 请求。
2. 58 模型大规模实验中，Prism 用 **16 GPU** 达到接近 99% TTFT SLO，而 MuxServe++ 需要 **32 GPU**；1B–8B、14B、70B 级模型激活分别约 **0.7 s、1.3 s、1.5 s**。
3. 生产 shadow replay 中，公司 A 的单 GPU token throughput 平均提高 **3.89×**且无新增 SLO 违约，公司 B 的单 GPU 收入提高 **2.86×**；`kvcached` 已部署于 **10K+ GPU**。

## 备注

- 最坏的恒定负载下，相比静态划分仍有最高 **4% TTFT / 13% TPOT** 管理开销。
- 论文基线中的 MuxServe++ 是作者移植到 SGLang 并接入 `kvcached` 的版本，不等同于原始实现；生产结果未公开具体模型组合和绝对延迟。
- 本条缩写使用 `PrismBalloon`，避免与仓库已有的 scheduling-memory co-design 论文 `PRISM` 冲突。
