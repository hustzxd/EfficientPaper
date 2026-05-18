# EfficientPaper

[![GitHub Stars](https://img.shields.io/github/stars/hustzxd/EfficientPaper?style=social)](https://github.com/hustzxd/EfficientPaper)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/hustzxd/EfficientPaper)](https://github.com/hustzxd/EfficientPaper)
[![Papers](https://img.shields.io/badge/papers-450+-blue)](https://github.com/hustzxd/EfficientPaper)

A curated collection of **450+** research papers on **Pruning**, **Quantization**, and **Efficient Inference/Training** for large language models and deep neural networks.

<p align="center">
<img src="/images/efficient_paper.png" width="800" title="EfficientPaper">
</p>

## Getting Started

```bash
git clone https://github.com/hustzxd/EfficientPaper
pip install protobuf==5.27.2 pandas arxiv
```

### MiMo API Key (Optional)

Adding papers can call Xiaomi MiMo LLM (`mimo-v2-flash`) to auto-generate Chinese summaries and keyword classification:

```bash
export MIMO_API_KEY="your-mimo-api-key"  # https://api.xiaomimimo.com/v1
```

> If not configured, auto-summarization is skipped — other features work normally.

### Quick Workflow

**1. Add paper from PDF:**

```bash
./add_paper_info.sh ~/Downloads/2512.01278v1.pdf
```

Extracts metadata, generates summary via MiMo, creates `.prototxt` and note files.

**2. Edit in browser:**

```bash
./start_editor.sh
```

Opens MkDocs (port 8000) + Editor API (port 8001). Visit `http://localhost:8000` to find, edit, and save papers.

**3. Deploy:**

```bash
./refresh_and_upload.sh 'update_paper_info'
```

Regenerates search data, commits, pushes, and deploys to GitHub Pages.

## Editor Features

| Category | Description |
|:---------|:------------|
| **Paper Info** | Title, abbreviation, URL, authors, institutions |
| **Publication** | Venue (arXiv, ICML, NeurIPS, ICLR, CVPR, ACL, ...) + year |
| **Code** | Repository URL with auto GitHub stars badge |
| **Keywords** | Multi-select: Quantization, Pruning, KV Cache, Sparsity, ... |
| **Cover** | Upload image, auto-saved and auto-referenced in `note.md` |
| **Baselines** | Format `year/abbr` with smart auto-complete |

## File Structure

```
meta/{year}/{paper_id}.prototxt   # Paper metadata
notes/{year}/{paper_id}/          # note.md + cover image
docs/js/papers.json               # Search index
scripts/paper_editor_server.py    # Editor backend
scripts/generate_search_data.py   # Search data generator
```

## Contributing

Contributions are welcome! To add a paper:

1. Fork this repo
2. Run `./add_paper_info.sh <paper.pdf>` to generate metadata
3. Run `./start_editor.sh` and edit paper details in the web UI
4. Submit a Pull Request

## Conference Timeline

<p align="center">
<img src="/notes/conference_timeline.png" width="800" title="Conference Timeline">
</p>

## 招聘

如果您对论文涉及到的研究内容感兴趣，同时有求职意向（[实习生/校招/社招](https://m.zhipin.com/gongsi/job/dc8e21b748a34c331HZz3Nu-GFU~.html?ka=m_seo_companys_all_jobs_boss)），可以发送简历到 zhaoxiandong27@gmail.com，欢迎沟通交流。

## References

1. [Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) [![Stars](https://img.shields.io/github/stars/Xnhyacinth/Awesome-LLM-Long-Context-Modeling?style=social)](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)
2. [Awesome-Efficient-Arch](https://github.com/weigao266/Awesome-Efficient-Arch) [![Stars](https://img.shields.io/github/stars/weigao266/Awesome-Efficient-Arch?style=social)](https://github.com/weigao266/Awesome-Efficient-Arch)
3. [Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) [![Stars](https://img.shields.io/github/stars/horseee/Awesome-Efficient-LLM?style=social)](https://github.com/horseee/Awesome-Efficient-LLM)
4. [Awesome-Diffusion-Inference](https://github.com/DefTruth/Awesome-Diffusion-Inference) [![Stars](https://img.shields.io/github/stars/DefTruth/Awesome-Diffusion-Inference?style=social)](https://github.com/DefTruth/Awesome-Diffusion-Inference)
5. [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference) [![Stars](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference?style=social)](https://github.com/DefTruth/Awesome-LLM-Inference)
6. [LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) [![Stars](https://img.shields.io/github/stars/AmberLJC/LLMSys-PaperList?style=social)](https://github.com/AmberLJC/LLMSys-PaperList)
7. [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) [![Stars](https://img.shields.io/github/stars/Hannibal046/Awesome-LLM?style=social)](https://github.com/Hannibal046/Awesome-LLM)
8. [Awesome-LLM-System-Papers](https://github.com/AmadeusChan/Awesome-LLM-System-Papers) [![Stars](https://img.shields.io/github/stars/AmadeusChan/Awesome-LLM-System-Papers?style=social)](https://github.com/AmadeusChan/Awesome-LLM-System-Papers)
9. [compiler-and-arch](https://github.com/KnowingNothing/compiler-and-arch) [![Stars](https://img.shields.io/github/stars/KnowingNothing/compiler-and-arch?style=social)](https://github.com/KnowingNothing/compiler-and-arch)
10. [PaperCopilot](https://papercopilot.com/paper-list)
11. [Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) [![Stars](https://img.shields.io/github/stars/TreeAI-Lab/Awesome-KV-Cache-Management?style=social)](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)
12. [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) [![Stars](https://img.shields.io/github/stars/October2001/Awesome-KV-Cache-Compression?style=social)](https://github.com/October2001/Awesome-KV-Cache-Compression)
13. [Awesome-Pruning](https://github.com/he-y/Awesome-Pruning) [![Stars](https://img.shields.io/github/stars/he-y/Awesome-Pruning?style=social)](https://github.com/he-y/Awesome-Pruning)
14. [awesome-model-quantization](https://github.com/htqin/awesome-model-quantization) [![Stars](https://img.shields.io/github/stars/htqin/awesome-model-quantization?style=social)](https://github.com/htqin/awesome-model-quantization)
15. [Awesome-Deep-Neural-Network-Compression](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression) [![Stars](https://img.shields.io/github/stars/csyhhu/Awesome-Deep-Neural-Network-Compression?style=social)](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression)
16. [Efficient-Deep-Learning](https://github.com/AojunZhou/Efficient-Deep-Learning) [![Stars](https://img.shields.io/github/stars/AojunZhou/Efficient-Deep-Learning?style=social)](https://github.com/AojunZhou/Efficient-Deep-Learning)
17. [Model-Compression-Papers](https://github.com/chester256/Model-Compression-Papers) [![Stars](https://img.shields.io/github/stars/chester256/Model-Compression-Papers?style=social)](https://github.com/chester256/Model-Compression-Papers)
