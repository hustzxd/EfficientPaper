# TorchAO: PyTorch-Native Training-to-Serving Model Optimization

> Andrew Or, Apurva Jain, Daniel Vega-Myhre, Jesse Cai, Charles David Hernandez, Zhenrui Zheng, Driss Guessous, Vasiliy Kuznetsov, Christian Puhrsch, Mark Saroufim, Supriya Rao, Thien Tran, Aleksandar Samardžić

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

We present TorchAO, a PyTorch-native model optimization framework leveraging
quantization and sparsity to provide an end-to-end, training-to-serving
workflow for AI models. TorchAO supports a variety of popular model
optimization techniques, including FP8 quantized training, quantization-aware
training (QAT), post-training quantization (PTQ), and 2:4 sparsity, and
leverages a novel tensor subclass abstraction to represent a variety of
widely-used, backend agnostic low precision data types, including INT4, INT8,
FP8, MXFP4, MXFP6, and MXFP8. TorchAO integrates closely with the broader
ecosystem at each step of the model optimization pipeline, from pre-training
(TorchTitan) to fine-tuning (TorchTune, Axolotl) to serving (HuggingFace, vLLM,
SGLang, ExecuTorch), connecting an otherwise fragmented space in a single,
unified workflow. TorchAO has enabled recent launches of the quantized Llama
3.2 1B/3B and LlamaGuard3-8B models and is open-source at
https://github.com/pytorch/ao/.
