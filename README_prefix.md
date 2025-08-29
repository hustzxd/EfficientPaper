# EfficientPaper
Pruning, Quantization and efficient-inference/training paper list.

## Table of Contents
- [EfficientPaper](#efficientpaper)
  - [Getting Started](#getting-started)
  - [Paper List](#paper-list)
    - [keyword](cls_keyword.md)
    - [year](cls_year.md)
    - [publication](cls_publication.md)
    - [institution](cls_institution.md)
    - [author](cls_author.md)
  - [References](#references)



## Getting Started
```bash
git clone https://github.com/hustzxd/EfficientPaper
pip install protobuf==5.27.2 pandas arxiv 
```
1. Add paper information by `./add_paper_info.sh`
2. Run `./refresh_readme.sh`

<details><summary><b>efficient_paper.prototxt</b></summary>	
<p>

```
paper {
  title: "EfficientPaper: manage your research papers in an efficient way."
  abbr: "EfficientPaper"
  url: "https://github.com/hustzxd/EfficientPaper"
  authors: "hustzxd"
}
pub {
  where: "GitHub"
  year: 2023
}
code {
  type: "Pytorch"
  url: "https://github.com/hustzxd/EfficientPaper"
}
note {
  url: "EfficientPaper.md"
}
keyword {
  words: efficient_paper
}
```

</p>
</details>

<p align="center">
<img src="notes//conference_timeline.png" width="800" title="blank">
</p>
