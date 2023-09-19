# EfficientPaper
Pruning, Quantization and efficient-inference/training paper list.

> Hi there,
> 
> To make it easier for everyone to manage the paper list and contribute paper information to the repository, we have implemented a template. This template allows us to collect the necessary information for a paper in a structured manner. Once the information is added, the repository will automatically refresh the readme file to display the paper list, categorized by year, authors, and other relevant criteria. This way, it becomes effortless to maintain an organized and up-to-date paper collection.

## Table of Contents
- [EfficientPaper](#efficientpaper)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)

</p>
</details>

## Getting Started
1. Add paper information by `./add_paper_info.sh` or  `./add_paper_info.sh <name>`
2. Run `./refresh_readme.sh`

<details><summary><b>sparsegpt.prototxt</b></summary>	
<p>

```
paper {
  title: "SparseGPT: Massive Language Models Can be Accurately Pruned in one-shot."
  abbr: "SparseGPT"
  url: "https://arxiv.org/pdf/2301.00774.pdf"
  authors: "Elias Frantar"
  authors: "Dan Alistarh"
  institutions: "IST Austria"
  institutions: "Neural Magic"
}
pub {
  where: "arXiv"
  year: 2023
}
code {
  type: "Pytorch"
  url: "https://github.com/IST-DASLab/sparsegpt"
}
note {
  url: "SparseGPT.md"
}
keyword {
  words: "sparsity"
}
```

</p>
</details>


