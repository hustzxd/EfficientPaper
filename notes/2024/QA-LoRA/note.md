# QA-LoRA

<p align="center">
<img src="qalora.jpg" width="600" title="blank">
</p>

QLoRA主要用于减少finetuning时的memory cost，相较于LoRA，它的性能没有优势，但是QLoRA在进行推理时，有需要把AB两个高精度表示的矩阵融合到低位宽的权重中，导致最终融合的权重表示为高位宽，并不能满足量化的约束。


<p align="center">
<img src="IMG_0448.PNG" width="600" title="blank">
</p>

实验结果对比，比QLoRA差一点，也算正常
<p align="center">
<img src="qa_lora_result.jpg" width="600" title="blank">
</p>
