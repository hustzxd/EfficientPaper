# Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity

<p align="center">
<img src="fig5.png" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs) have demonstrated strong capabilities across
various domains, with recent advancements in challenging reasoning tasks such
as mathematics and programming. However, solving reasoning tasks often requires
long decoding chains (of thoughts), which incur $O(N)$ time and memory
consumption, where $N$ is the chain length. To mitigate $O(N)$ time and memory
consumption, existing sparsity-based algorithms propose retaining only the most
critical token's intermediate data (i.e., key-value cache) and discarding the
rest. However, these existing algorithms struggle with the ``impossible
trinity'' of accuracy, time, and memory. For example, the state-of-the-art
algorithm, Quest, achieves high accuracy with $O(L)$ time but $O(N)$ memory
($L$ is the cache budget, $L \ll N$). To address this issue, in this paper, we
identify a new attention pattern during the decode stage of reasoning tasks,
where milestone tokens (analogous to lemmas in mathematical proofs) emerge, are
utilized, and then become unimportant afterward. Based on this pattern, we
propose a new algorithm named RaaS that identifies and retains milestone tokens
only until they are no longer needed, achieving high accuracy with $O(L)$ time
and $O(L)$ memory complexity.
