# VORTA: Efficient Video Diffusion via Routing Sparse Attention

> Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Zhao Jin, Jingyi Liao, Shunyu Liu, Dacheng Tao

![111](fig6.png)

## Abstract

Video diffusion transformers have achieved remarkable progress in high-quality video generation, but remain computationally expensive due to the quadratic complexity of attention over high-dimensional video sequences. Recent acceleration methods enhance the efficiency by exploiting the local sparsity of attention scores; yet they often struggle with accelerating the long-range computation. To address this problem, we propose VORTA, an acceleration framework with two novel components: 1) a sparse attention mechanism that efficiently captures long-range dependencies, and 2) a routing strategy that adaptively replaces full 3D attention with specialized sparse attention variants. VORTA achieves an end-to-end speedup $1.76\times$ without loss of quality on VBench. Furthermore, it can seamlessly integrate with various other acceleration methods, such as model caching and step distillation, reaching up to speedup $14.41\times$ with negligible performance degradation. VORTA demonstrates its efficiency and enhances the practicality of video diffusion transformers in real-world settings. Codes and weights are available at https://github.com/wenhao728/VORTA.
