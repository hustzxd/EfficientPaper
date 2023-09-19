# Knowledge-preserving Pruning for Pre-trained Language Models without Retraining

> This is a retraning-free structured pruning approach.

<p align="center">
  <img src="./kp.jpg" width="600" title="kp">
</p>

## Method
- Key idea
  - **Selecting pruning targets**
    - Neurons and attention heads that minimally reduce the PLM’s knowledge
  - **Iterative pruning**
    - Use knowledge reconstruction for each sub-layer to handle the distorted inputs by pruning.
- K-pruning (Knowledge-preserving pruning)
  - knowledge measurement
  - knowledge-preserving mask search
  - knowledge-preserving pruning