# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

> Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Speculative decoding accelerates LLM inference by decoupling draft generation from target verification, but long parallel draft blocks face suffix acceptance decay, while fixed-length verification wastes batch capacity under high concurrency. DSpark combines a semi-autoregressive drafter with confidence-scheduled verification: a parallel backbone produces draft hidden states, a lightweight sequential head restores intra-block dependency, and a calibrated confidence scheduler dynamically chooses per-request verification length using engine throughput profiles. On offline benchmarks, DSpark improves accepted length over Eagle3 and DFlash; in DeepSeek-V4 live serving, it accelerates per-user generation by 60%-85% on V4-Flash and 57%-78% on V4-Pro at matched throughput levels.

## 一句话总结

DSpark 把 speculative decoding 从“固定多猜几个 token”推进到“按置信度和系统负载动态分配验证预算”，在 DeepSeek-V4 生产流量中同时提升单用户 TPS 和高并发吞吐边界。

## 创新点

1. 半自回归 draft 架构：保留 DFlash 式并行 backbone 的单次前向优势，只在输出侧加轻量 Markov/RNN sequential head，让后续 draft token 条件化于已采样前缀，缓解 parallel drafter 的 suffix decay。
2. 置信度调度验证：为每个 draft 位置预测条件存活概率，并用 Sequential Temperature Scaling 校准累计 prefix survival probability，使验证长度选择不再依赖静态阈值。
3. 硬件感知 prefix scheduler：把每个请求的待验证 token 按期望边际收益排序，并结合在线 engine 的 SPS(batch size) 曲线选择全局验证预算；生产实现进一步适配 ZOS/CUDA graph 和 variable-length query execution。

## 带来什么提升

1. 离线 accepted length 明显提升：在 Qwen3-4B/8B/14B 上，相对 Eagle3 macro-average accepted length 分别提升 30.9%/26.7%/30.0%，相对 DFlash 提升 16.3%/18.4%/18.3%。
2. 更长 draft block 仍保持有效：当 proposal length 增大到 15，DSpark 相对 DFlash 在 math/code/chat 上的 accepted length 增益扩大到 30%/26%/22%，而 sequential head 带来的整轮 latency overhead 仅约 0.2%-1.3%。
3. 生产 serving frontier 外移：在 DeepSeek-V4-Flash live traffic 中，80 tok/s/user SLA 下 aggregate throughput 提升 51%，120 tok/s/user 高交互 SLA 下维持 MTP-1 难以支撑的并发；V4-Pro 在 35 tok/s/user 下吞吐提升 52%，匹配吞吐时单用户速度提升 57%-78%。

## 备注

1. DSpark 的核心价值不只是 drafter 更准，而是把 acceptance probability、verification budget 和系统 batch capacity 统一进 serving scheduler；这比单纯增加 MTP heads 更适合高并发生产环境。
2. 论文也指出固定 draft-side cost 仍存在：对天然低 acceptance 的复杂请求，先生成完整 draft block 的成本可能无法回收，后续可考虑 difficulty-aware early exit 或跳过 full-block draft。
