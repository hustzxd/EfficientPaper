# Accelerating Model Loading in LLM Inference by Programmable Page Cache

> Yubo Liu, Hongbo Li, Xiaojia Huang, Yongfeng Wang, Hanjun Guo, Hui Chen, Yuxin Ren, Ning Jia | Huawei Technologies Co., Ltd. | FAST 2026

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

论文针对 LLM 服务启动时模型权重从 SSD 加载到加速器的瓶颈，提出可编程页缓存框架 PPC，以及基于 PPC 的加载策略 MAIO。它在保留现有文件系统和推理框架接口的同时，按服务配置记录 I/O 模板，联合优化预取、NUMA 放置和页淘汰，提高模型加载效率。

## 一句话总结

把模型加载的访问顺序、目标 XPU 和数据生命周期交给可编程页缓存策略，实现更快且内存可控的推理服务启动。

## 创新点

1. **PPC 框架**：内核侧只读 Routing File System 堆叠于现有文件系统，通过非阻塞 userspace procedure call 上报 read/mmap 的 cache miss；用户态 Cache Policy Runtime 执行可动态注册的预取与淘汰策略，并支持目录级部署。
2. **服务级 I/O 模板**：MAIO 以模型、TP 和 PD 等运行配置构成 service ID，记录各 XPU worker 的文件、offset、size 序列；相同配置复用模板。规格变化后由 MaaS 控制面重新生成，模板并非对任意运行配置通用。
3. **三项联合策略**：可中断预取依据最新 miss 位置跳转，缓解落后预取与前台 I/O 的竞争；XPU affinity loading 将页放在目标设备对应的 NUMA node；Burn-after-Reading 按访问进度回收已加载权重对应的主机页，并保留默认 **1 GB**安全距离减少抖动。

## 带来什么提升

1. **模型加载**：Ascend 910B2、Linux 5.10、vLLM-Ascend 0.9.2 上测试五个 Qwen/Llama 模型；内存充足时，相对 Native 加载延迟最高降低 **79%**，相对 SLLM-NPU 在大模型场景最高降低 **17%**。将加载可用内存限制为 **64 GB** 时，相对测试方案最高降低 **74%**。
2. **服务启动与弹性吞吐**：整体 startup 延迟在内存充足/受限时分别最高降低 **38% / 51%**。MaaS 交替部署实验中，相对 Native 的全流程平均 token 吞吐分别最高提升 **13% / 28%**，来自缩短启动空窗；服务持续运行越久，收益越小。
3. **真实部署与开销**：两节点、16 个 Ascend 910B3 上的 DeepSeek-R1-671B 从本地 SSD 冷加载由 **649 秒降至 452 秒**。PPC 在 EXT4/XFS 的 memcpy-after-mmap 微基准中分别最多引入 **3.7% / 6.4%**额外开销，框架内存约 **30 MB**；该模型的 I/O 模板只有 **545 KB**。

## 备注

- 本文优化的是**模型权重加载**，没有提出在线 KV Cache 管理算法。PPC 是框架，MAIO 是具体策略；openEuler 的 ModelFS 是对应的工程实现，不应与库中 PPCL 论文混淆。
- “非侵入式”仍需要安装独立内核模块并切换访问路径；当前 RFS 只支持只读场景。BAR 根据 I/O 进度推断页已使用，并未直接观测全部 host-to-device 传输完成；默认安全距离是经验参数。
- 主要性能证据来自 Ascend 平台；SLLM-NPU 对照因适配限制仅在内存充足、Transformers 下运行。BlitzScale 未参与性能对比。摘要的“最高 36%”使用更广的测试方案口径，不能替代正文相对 Native 的 **13% / 28%**。
- [论文原文](https://www.usenix.org/system/files/fast26-liu-yubo.pdf) · [会议页面](https://www.usenix.org/conference/fast26/presentation/liu-yubo) · [ModelFS 内核代码](https://gitee.com/openeuler/kernel/tree/OLK-6.6/fs/mfs)（由 [openEuler 官方方案文档](https://gitee.com/openeuler/llm_solution/blob/master/README_en.md)确认的工程实现入口，并非论文全部实验 artifact）。
