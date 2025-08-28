# Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

![](tab1.png)

## Abstract

Key-Value cache (\texttt{KV} \texttt{cache}) compression has emerged as a
promising technique to optimize Large Language Model (LLM) serving. It
primarily decreases the memory consumption of \texttt{KV} \texttt{cache} to
reduce the computation cost. Despite the development of many compression
algorithms, their applications in production environments are still not
prevalent. In this paper, we revisit mainstream \texttt{KV} \texttt{cache}
compression solutions from a practical perspective. Our contributions are
three-fold. First, we comprehensively review existing algorithmic designs and
benchmark studies for \texttt{KV} \texttt{cache} compression and identify
missing pieces in their performance measurement, which could hinder their
adoption in practice. Second, we empirically evaluate representative
\texttt{KV} \texttt{cache} compression methods to uncover two key issues that
affect the computational efficiency: (1) while compressing \texttt{KV}
\texttt{cache} can reduce memory consumption, current implementations (e.g.,
FlashAttention, PagedAttention) do not optimize for production-level LLM
serving, resulting in suboptimal throughput performance; (2) compressing
\texttt{KV} \texttt{cache} may lead to longer outputs, resulting in increased
end-to-end latency. We further investigate the accuracy performance of
individual samples rather than the overall performance, revealing the intrinsic
limitations in \texttt{KV} \texttt{cache} compression when handling specific
LLM tasks. Third, we provide tools to shed light on future \texttt{KV}
\texttt{cache} compression studies and facilitate their practical deployment in
production. They are open-sourced in
\href{https://github.com/LLMkvsys/rethink-kv-compression}{https://github.com/LLMkvsys/rethink-kv-compression}.

- KV cache压缩后缺乏性能的评估
- PagedAttention FalshAttention没有对KV 压缩进行适配
- KV 压缩后可能导致output变长，从而增加end-to-end的latency