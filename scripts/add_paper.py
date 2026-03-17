import argparse
import datetime
import hashlib
import os
import random
import re
import string
import sys

import arxiv
import google.protobuf as pb
import google.protobuf.text_format
from openai import OpenAI

# Add project root to path before importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import efficient_paper_pb2 as eppb


NOTE_SUMMARY_PROMPT = """你是一个LLM效率优化领域的论文摘要专家。

根据论文的标题和摘要，用中文写3-5句话的概要性总结，包括：
1. 这篇论文要解决什么问题
2. 提出了什么方法/技术
3. 取得了什么效果/提升

要求简洁准确，突出核心贡献。只返回总结文本，不要其他内容。"""


# Map display name -> proto enum name
KEYWORD_MAP = {
    "Sparse/Pruning": "sparse_pruning",
    "Sparsity (Attention)": "attention_sparsity",
    "Sparsity (Activation)": "activation_sparsity",
    "Sparsity (Weight)": "weight_sparsity",
    "Sparsity (Structured)": "structured_sparsity",
    "Quantization": "quantization",
    "Quantization (KV Cache)": "kv_cache_quant",
    "Sparse/Eviction (KV Cache)": "kv_cache_sparse",
    "KV Cache Management": "kv_cache_management",
    "Comm-Comp Overlap": "overlap",
    "Speculative Decoding": "speculative_decoding",
    "Performance Modeling": "performance_modeling",
    "LLM Deployment": "deployment",
    "Survey": "survey",
    "Network Structure Design": "structure_design",
    "Low Rank Decomposition": "low_rank",
    "Layer Fusion (Reduce IO)": "fusion",
    "Efficient Training": "efficient_training",
    "Tool": "tool",
    "Benchmark": "benchmark",
    "Kernel Generation": "kernel_generation",
}

KEYWORD_CATEGORIES = list(KEYWORD_MAP.keys())

KEYWORD_PROMPT = f"""你是一个LLM效率优化领域的论文分类专家。

根据论文的标题和摘要，从以下关键词类别中选择1-2个最相关的类别：
{', '.join(KEYWORD_CATEGORIES)}

规则：
- 只选择非常确定相关的类别，不确定就不选
- 返回格式：每行一个关键词，使用上面列表中的原始英文名称
- 如果没有任何类别匹配，只返回 "none"
- 不要返回其他任何内容"""


def _get_llm_client():
    return OpenAI(
        api_key=os.environ.get("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1",
    )


def generate_note_summary(title, abstract):
    """Use MiMo to generate a Chinese summary for a paper note."""
    try:
        client = _get_llm_client()
        resp = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[
                {"role": "system", "content": NOTE_SUMMARY_PROMPT},
                {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
            ],
            max_completion_tokens=1024,
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[93m⚠ Failed to generate MiMo summary: {e}\033[0m")
        return None


def generate_keywords(title, abstract):
    """Use MiMo to select 1-2 keyword categories for a paper.
    Returns list of (display_name, enum_name) tuples, or None."""
    try:
        client = _get_llm_client()
        resp = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[
                {"role": "system", "content": KEYWORD_PROMPT},
                {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
            ],
            max_completion_tokens=256,
            temperature=0.1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content.strip()
        if raw.lower() == "none":
            return None
        # Validate against known categories and map to enum names
        keywords = []
        for line in raw.split('\n'):
            line = line.strip().strip('-').strip()
            if line in KEYWORD_MAP:
                keywords.append((line, KEYWORD_MAP[line]))
        return keywords if keywords else None
    except Exception as e:
        print(f"\033[93m⚠ Failed to generate keywords: {e}\033[0m")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper INFO")
    parser.add_argument("-f", "--file", default=None, help="The file name")
    args = parser.parse_args()
    return args


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def extract_code_url(summary):
    """Extract GitHub/code repository URL from paper summary.

    Args:
        summary: Paper abstract/summary text

    Returns:
        str: Code repository URL if found, None otherwise
    """
    if not summary:
        return None

    # Common patterns for code repositories
    patterns = [
        r'https?://github\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://huggingface\.co/[\w\-\.]+/[\w\-\.]+',
        r'https?://gitee\.com/[\w\-\.]+/[\w\-\.]+',
    ]

    for pattern in patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            url = match.group(0)
            # Remove trailing punctuation
            url = re.sub(r'[.,;:\)]+$', '', url)
            return url

    return None


def extract_institutions(authors):
    """Extract institutions from arXiv author objects.

    Args:
        authors: List of arxiv.Result.Author objects

    Returns:
        list: List of unique institution names
    """
    institutions = []
    seen = set()

    for author in authors:
        # Try to get affiliation from author object
        if hasattr(author, 'affiliation') and author.affiliation:
            affiliation = author.affiliation.strip()
            if affiliation and affiliation not in seen:
                institutions.append(affiliation)
                seen.add(affiliation)

    return institutions


def main():
    args = parse_args()

    if args.file is not None:
        arxiv_id = args.file.split("/")[-1].replace(".pdf", "")
    else:
        arxiv_id = input("Please input the abs of arXiv (Default: None) \n arxiv_id: ")
    if len(arxiv_id) != 0:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    else:
        paper = None

    if paper is not None:
        print("Paper title from arxiv:")
        print(f"\033[96m<<< {paper.title} >>>\033[0m")
        if args.file is not None and os.path.exists(args.file):
            root_dir = os.path.dirname(args.file)
            new_name = paper.title.replace(" ", "_") + ".pdf"
            new_name = new_name.replace(":", "_")
            new_name_path = os.path.join(root_dir, new_name) if root_dir else new_name
            os.rename(args.file, new_name_path)
            print(f"{args.file} => {new_name_path}")

    name = input("Please add short name for a paper (Default: random code) \n abbr name: ")
    if len(name) == 0:
        name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    print("Paper name is: " + name)

    pinfo = eppb.PaperInfo()
    with open("proto/template.prototxt", "r") as rf:
        pb.text_format.Merge(rf.read(), pinfo)

    if paper is not None:
        pinfo.paper.title = paper.title
        pinfo.paper.abbr = name
        pinfo.paper.url = paper.entry_id
        pinfo.update_time = int(datetime.datetime.now().timestamp())
        authors = [author.name for author in paper.authors]
        try:
            pinfo.paper.authors.clear()
        except:
            pinfo.paper.authors.pop()
            pinfo.paper.authors.pop()
        pinfo.paper.authors.extend(authors)
        pinfo.pub.year = paper.published.year

        # Automatically extract institutions
        institutions = extract_institutions(paper.authors)
        if institutions:
            print(f"\033[92m✓ Found {len(institutions)} institution(s):\033[0m")
            for inst in institutions:
                print(f"  - {inst}")
            try:
                pinfo.paper.institutions.clear()
            except:
                pass
            pinfo.paper.institutions.extend(institutions)
        else:
            print("\033[93m⚠ No institutions found in arXiv metadata\033[0m")

        # Automatically extract code repository URL
        code_url = extract_code_url(paper.summary)
        if code_url:
            print(f"\033[92m✓ Found code repository: {code_url}\033[0m")
            pinfo.code.url = code_url
        else:
            print("\033[93m⚠ No code repository URL found in abstract\033[0m")

        # Use MiMo to select keywords
        print("\033[96mSelecting keywords via MiMo...\033[0m")
        keywords = generate_keywords(paper.title, paper.summary)
        if keywords:
            display_names = [k[0] for k in keywords]
            enum_names = [k[1] for k in keywords]
            print(f"\033[92m✓ Keywords: {', '.join(display_names)}\033[0m")
            while len(pinfo.keyword.words):
                pinfo.keyword.words.pop()
            Word = eppb.Keyword.Word
            for enum_name in enum_names:
                pinfo.keyword.words.append(Word.Value(enum_name))
        else:
            print("\033[93m⚠ No matching keywords, keeping default 'none'\033[0m")
    else:
        current_year = datetime.date.today().year
        year = input(f"Please input the public year of this paper (Default: {current_year}) \n Year: ")
        if len(year) != 0:
            assert 1990 <= int(year) <= 2030
            year = int(year)
        else:
            year = 2025
        pinfo.pub.year = year
        pinfo.update_time = int(datetime.datetime.now().timestamp())

    root_dir = "./"
    # Ensure year-based directory exists
    year_dir = os.path.join(root_dir, "meta", str(pinfo.pub.year))
    os.makedirs(year_dir, exist_ok=True)

    # Check if file already exists
    new_path = os.path.join(year_dir, f"{name}.prototxt")
    old_path = os.path.join(root_dir, "meta", f"{name}.prototxt")

    if os.path.exists(new_path) or os.path.exists(old_path):
        print("The file `{}` already exists, please use another name".format(name))
        return

    with open(new_path, "w") as wf:
        print(pinfo)
        print("Writing paper information into {}/meta/{}/{}.prototxt".format(root_dir, pinfo.pub.year, name))
        wf.write(str(pinfo))

    # mkdir note with year-based structure
    note_dir = f"notes/{pinfo.pub.year}/{name}"
    os.makedirs(note_dir, exist_ok=True)
    note_file = os.path.join(note_dir, "note.md")

    if not os.path.exists(note_file):
        if paper is not None:
            title = paper.title
            summary = paper.summary
        else:
            title = name
            summary = ""
        note_content = f"""# {title}\n\n"""
        if paper is not None:
            note_content += f"""> {", ".join(authors)}\n\n"""
        # note_content += """<p align="center">\n<img src="../../blank.jpg" width="600" title="blank">\n</p>\n\n"""
        note_content += f"""![111](../../blank.jpg)\n\n"""
        note_content += f"""## Abstract\n\n{summary}\n"""

        # Generate MiMo summary
        if summary:
            print("\033[96mGenerating MiMo summary...\033[0m")
            mimo_summary = generate_note_summary(title, summary)
            if mimo_summary:
                note_content += f"\n\n---\n\n*以下总结由 MiMo 生成：*\n\n{mimo_summary}\n"
                print(f"\033[92m✓ MiMo summary generated\033[0m")

        with open(note_file, "w") as wf:
            wf.write(note_content)


if __name__ == "__main__":
    main()
