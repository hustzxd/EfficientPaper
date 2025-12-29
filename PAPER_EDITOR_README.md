# Paper Metadata Visual Editor

一个可视化的论文元数据编辑器,用于编辑 prototxt 文件。

## 功能特性

- 可视化表单界面编辑 prototxt 文件
- 支持所有字段类型:文本、数字、URL、数组、下拉列表
- 关键词使用多选框方式选择
- 发布会议提供预设选项
- 实时保存到 prototxt 文件
- 与 MkDocs 同时运行,无冲突

## 快速开始

### 一键启动(推荐)

```bash
./start_editor.sh
```

这个脚本会同时启动:
- **MkDocs** (端口 8000) - 提供搜索页面和静态文件
- **Paper Editor API** (端口 8001) - 提供编辑功能的 API

### 手动启动

如果需要分别启动服务:

**终端 1 - 启动 MkDocs:**
```bash
mkdocs serve
```

**终端 2 - 启动编辑 API:**
```bash
python scripts/paper_editor_server.py
```

## 使用方法

1. **访问搜索页面**
   ```
   http://localhost:8000/search/
   ```

2. **查找要编辑的论文**
   - 使用搜索框或筛选器找到论文
   - 点击论文卡片上的 **"Edit"** 链接

3. **编辑论文元数据**
   - 在可视化表单中修改内容
   - 支持动态添加/删除作者、机构等
   - 关键词使用复选框多选

4. **保存更改**
   - 点击 **"Save Changes"** 按钮
   - 系统会自动保存到 prototxt 文件
   - 保存成功后会跳转回搜索页面

5. **更新搜索数据**
   - 修改后需要重新生成 papers.json:
   ```bash
   PYTHONPATH=$PWD python scripts/generate_search_data.py
   ```
   - 或运行:
   ```bash
   ./weekly_paper.sh
   ```

### 4. 字段说明

#### Paper Information (论文信息)
- **Title**: 论文标题 (必填)
- **Abbreviation**: 简称/缩写
- **Paper URL**: 论文链接 (通常是 arXiv 链接)
- **Authors**: 作者列表 (可添加多个)
- **Institutions**: 机构列表 (可添加多个)

#### Publication (发表信息)
- **Venue**: 发表会议/期刊 (下拉选择)
  - arXiv, ICML, NeurIPS, ICLR, CVPR, ICCV, ECCV, ACL, EMNLP, NAACL, AAAI, IJCAI, SIGMOD, KDD
- **Year**: 发表年份 (必填)

#### Code (代码)
- **Code Type**: 代码框架 (PyTorch, TensorFlow, JAX, Others)
- **Code Repository URL**: 代码仓库链接

#### Keywords (关键词)
使用复选框选择适用的关键词:
- None
- Attention Sparsity
- Activation Sparsity
- Weight Sparsity
- Structured Sparsity
- Sparse Pruning
- KV Cache Quantization
- Quantization
- Overlap
- Performance Modeling
- Deployment
- Survey
- Structure Design
- Low Rank
- KV Cache
- Fusion
- Efficient Training
- Tool
- Benchmark

#### Cover Image (封面图)

- **Cover Image Filename**: 封面图片文件名 (自动填充)
- **Upload Cover Image**: 点击 "Choose Image" 按钮上传图片
  - 支持的格式: PNG, JPG, JPEG, GIF, WebP
  - 图片会自动保存到 `notes/{year}/{paper_id}/` 目录
  - 文件名自动命名为 `cover.{ext}`
  - 上传成功后会显示预览
  - 如果论文已有封面图,会自动显示预览

#### Baseline Methods (基线方法)

- **Methods**: 基线方法列表 (可添加多个)
- **智能搜索**:
  - 开始输入时会自动显示匹配的现有方法
  - 使用 ↑↓ 键导航,Enter 选择
  - 点击下拉列表中的方法快速选择
  - 也可以直接输入新的方法名
  - 提示: "Type to search existing methods, or enter a new one"

## 工作流程

### 更新 papers.json

每次修改 prototxt 文件后,需要重新生成 `papers.json`:

```bash
PYTHONPATH=/Users/xiandong/projects/EfficientPaper:$PYTHONPATH python scripts/generate_search_data.py
```

或者创建一个便捷脚本:

```bash
# 添加到 weekly_paper.sh 或创建新的更新脚本
./weekly_paper.sh
```

## 架构说明

### 文件结构

```
EfficientPaper/
├── docs/
│   ├── search.html          # 搜索页面 (已更新,添加 Edit 链接)
│   ├── edit.html            # 编辑页面 (新增)
│   └── js/
│       └── papers.json      # 论文数据 (包含 prototxt_path 字段)
├── scripts/
│   ├── generate_search_data.py      # 生成搜索数据 (已更新)
│   └── paper_editor_server.py       # 编辑服务器 (新增)
├── meta/
│   └── {year}/
│       └── {paper_id}.prototxt      # 论文元数据文件
└── proto/
    ├── efficient_paper.proto        # Protobuf 定义
    └── efficient_paper_pb2.py       # 生成的 Python 代码
```

### API 端点

#### GET /api/load-paper?path={path}
加载指定路径的 prototxt 文件

**参数:**
- `path`: prototxt 文件路径,例如 `meta/2025/P0JBYHCN.prototxt`

**响应:**
```json
{
  "paper": {...},
  "pub": {...},
  "code": {...},
  "keyword": {...},
  "cover": {...},
  "baseline": {...}
}
```

#### POST /api/save-paper
保存论文元数据到 prototxt 文件

**请求体:**
```json
{
  "path": "meta/2025/P0JBYHCN.prototxt",
  "data": {
    "paper": {...},
    "pub": {...},
    ...
  }
}
```

**响应:**
```json
{
  "success": true,
  "message": "Paper saved successfully"
}
```

## 技术细节

- **前端**: 纯 HTML/CSS/JavaScript,无需构建工具
- **后端**: Python HTTP 服务器,使用 protobuf 解析和生成
- **数据格式**: Protocol Buffers (prototxt)
- **通信**: REST API (JSON)

## 注意事项

1. 编辑服务器仅用于本地开发,不应暴露到公网
2. 修改 prototxt 后记得重新生成 papers.json
3. 保存前确保所有必填字段已填写
4. 作者和机构可以动态添加/删除
5. 关键词至少选择一个 (可以选择 "None")
