# Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks

> Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, Vamse Kumar Subbiah

![111](../../blank.jpg)

## Abstract

Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearch Bench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 41-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. An ablation study across prompt sizes (500-50,000 tokens) and tool call counts (3-50) demonstrates universal linear cost and TTFT benefits, after the provider caching token minimum, and reveal provider-specific strategy discrepancies across variants. We provide nuanced discussion and guidance for implementing prompt caching in production agentic systems.


---

*以下总结由 MiMo 生成：*

这篇论文针对大型语言模型（LLM）代理在长时程多轮任务中因频繁工具调用导致成本和延迟增加的问题，评估了提示缓存的效果。研究者对三大LLM提供商（OpenAI、Anthropic、Google）的三种缓存策略进行了全面测试，并在DeepResearch Bench基准上测量了API成本和首词延迟。结果表明，提示缓存可降低41-80%的API成本并提升13-31%的首词速度，而通过智能控制缓存块（如将动态内容置于提示末尾）能获得更稳定收益，优于全上下文缓存。
