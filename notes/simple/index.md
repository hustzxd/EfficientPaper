# Structured Pruning for Efficient Generative Pre-trained Language Models

<p align="center">
  <img src="./cover.jpg" width="800" title="cover">
</p>

## Method

- Sparsity-induced Mask Learning
    - Teacher-Student KD loss and L1 regularization to optimize sparse mask.
    - The learnable sparse mask is initialized to 1, and is updated with gradient descent during training. After learning the mask, these masks are binarized according to a threshold determined by a given sparsity.

<p align="center">
  <img src="./eq5.jpg" width="250" title="cover">
</p>

- Fine-tuning
    - Fix sparse mask and finetune weights
    - KD Loss + Local KD loss

MSE loss of K, V
<p align="center">
  <img src="./eq6.jpg" width="350" title="cover">
</p>

$\ell_{hidden}$ is hidden state distillation loss:

<p align="center">
  <img src="./eq7.jpg" width="350" title="cover">
</p>

## Results

<p align="center">
  <img src="./res1.jpg" width="350" title="cover">
</p>

论文中没有给出具体的GPT2，按照模型大小推测应该是GPT2-small，ppl=29，所以论文中的结果可能在wikitext2上进行了finetune
<p align="center">
  <img src="./gpt2_results.jpg" width="750" title="cover">
</p>