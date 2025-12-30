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

### 环境准备

```bash
git clone https://github.com/hustzxd/EfficientPaper
pip install protobuf==5.27.2 pandas arxiv
```

### 工作流程

#### 第一步：添加论文信息

使用 `add_paper_info.sh` 脚本从 PDF 文件自动提取论文信息：

```bash
./add_paper_info.sh ~/Downloads/2512.01278v1.pdf
```

这个脚本会：
- 自动从 PDF 提取论文标题、作者、摘要等信息
- 在 `meta/YYYY/` 目录下创建对应的 `.prototxt` 文件
- 在 `notes/YYYY/{paper_id}/` 目录下创建笔记文件

#### 第二步：启动编辑器并编辑

运行编辑器服务：

```bash
./start_editor.sh
```

这个脚本会同时启动：
- **MkDocs 服务** (端口 8000) - 提供搜索和预览界面
- **Paper Editor API** (端口 8001) - 提供编辑功能的后端 API

然后访问搜索页面 `http://localhost:8000/search/`，在可视化界面中：
1. **查找论文** - 使用搜索框或筛选器找到刚添加的论文
2. **点击 Edit** - 点击论文卡片上的 "Edit" 链接
3. **编辑信息** - 在表单中完善论文信息（标题、作者、机构、关键词、封面图片、baseline methods 等）
4. **保存** - 点击 "Save Changes" 按钮并确认

#### 第三步：确认并部署到 GitHub

确认编辑无误后，使用 `refresh_readme.sh` 部署到 GitHub：

```bash
./refresh_readme.sh 'update paper info'
```

这个脚本会：
- 重新生成搜索数据和分类页面
- 提交所有更改并推送到 GitHub
- 自动触发 GitHub Pages 部署

> 详细使用说明请参考 [PAPER_EDITOR_README.md](PAPER_EDITOR_README.md)

<p align="center">
<img src="notes//conference_timeline.png" width="800" title="blank">
</p>


## 招聘

如果您对论文涉及到的研究内容感兴趣，同时有求职意向（[实习生/校招/社招](https://m.zhipin.com/gongsi/job/dc8e21b748a34c331HZz3Nu-GFU~.html?ka=m_seo_companys_all_jobs_boss)），可以发送简历到zhaoxiandong27@gmail.com，欢迎沟通交流。