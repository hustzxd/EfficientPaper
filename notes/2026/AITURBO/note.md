# Fast Cloud Storage for AI Jobs via Grouped I/O API with Transparent Read/Write Optimizations

> Yingyi Hao, Ting Yao, Xingda Wei, Dingyan Zhang, Tianle Sun, Yiwen Zhang, Zhiyong Fu, Huatao Wu, Rong Chen | Shanghai Jiao Tong University; Huawei Cloud | FAST 2026

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

AITURBO 面向训练 checkpoint、推理扩容和 KV Cache 读取等大块 I/O，利用计算节点的空闲 DRAM 与高速计算网络扩展有效存储带宽，再通过 grouped I/O API 向存储层提供一组请求的共同需求，自动生成去重、负载均衡和流水传输计划。系统已部署于 Huawei Cloud 的训练作业，论文另评估了推理场景。

## 一句话总结

将跨 XPU 的共同 I/O 需求暴露给存储层，结合 DRAM staging 和计算网络转发，减少重复访问与慢存储路径上的等待。

## 创新点

1. **Grouped I/O API**：应用显式指定参与读写的 client group，让 job controller 同时看到多个文件及其需求；写操作分别返回“进入 DRAM”和“后端持久化”两个 future，支持按应用容错要求选择继续执行的时机。
2. **透明去重与传输规划**：写路径通过 XPU 上的 BLAKE3 checksum 识别重复 chunk，并在带宽、容量和副本约束下规划来源与目的节点；稳定的 checkpoint 模式可复用去重元数据及计划，避免每轮重算。
3. **双网络协作**：读路径只从存储取一份共享数据，随后沿计算网络转发给各实例；已有的跨节点 DRAM 副本也可直接复用。tensor-native 文件布局减少序列化，P2P RDMA 降低通信初始化开销，分阶段传输通过流水交叠。

## 带来什么提升

1. **Checkpoint 写入**：最多 64 XPU 的训练实验中，相对 Megatron + SFSTURBO，checkpoint 时间加速约 **3.9–58.8 倍**；相对作者复现主要机制的 Gemini，最高 **5.9 倍**。主要收益来自内存 checkpoint、去重和负载均衡，不能当作同步持久化写入的等价比较。
2. **模型分发**：Qwen-72B 在 **64 XPU** 上分发只需 **2.25 秒**，前提是模型已经读取过且缓存于 host DRAM。冷读仍受后端存储限制：1 GB/s 配额下，135 GB 模型在 8 XPU 上约需 **173 秒**；优势在于随后通过计算网络将一份模型分发给更多实例。
3. **KV 读取与集成成本**：Qwen-14B、8 XPU、30 分钟 Qwen-Bailian trace 中，Mooncake 接入 AITURBO 后平均 TTFT 从 **0.85 秒降至 0.65 秒**，约降低 **23%**。Megatron 和 Mooncake 分别仅需约 **286 / 44 行**集成修改。

## 备注

- Mooncake 实验**没有使用 group API**，因为运行中推理实例难以同步；该结果验证的是后端与计算网络路径的优化，不能证明 grouped KV 去重的收益。对照采用特定 Mooncake commit，其从存储读回的 KV 不会重新进入 KV 缓存层。
- DRAM future 完成不代表持久化完成；全部内存副本在 flush 前丢失时，应用需要识别损坏文件并回退。适用性依赖可借用的 DRAM、计算网络带宽和大块传输，细粒度读取及复杂多租户隔离尚待研究。
- [论文原文](https://www.usenix.org/system/files/fast26-hao.pdf) · [会议页面](https://www.usenix.org/conference/fast26/presentation/hao)。未在论文或会议页面找到公开系统代码。
