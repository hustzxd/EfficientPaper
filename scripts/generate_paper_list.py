import argparse
import copy
import os
import sys

import google.protobuf as pb
import google.protobuf.text_format
import pandas as pd

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper INFO")
    parser.add_argument(
        "-d", "--detail", action="store_true", default=False, help="Whether to display information in a detail way."
    )
    args = parser.parse_args()
    return args


PUBLISH_COLOR = {
    # AI/ML Conferences (warm colors)
    "AAAI": "FF4500",  # Orange Red
    "ICLR": "FF6B6B",  # Coral Pink
    "ICML": "FF8C00",  # Dark Orange
    "NeurIPS": "FF1493",  # Deep Pink
    # NLP Conferences (cool colors)
    "ACL": "4169E1",  # Royal Blue
    "COLM": "6495ED",  # Cornflower Blue
    "Coling": "1E90FF",  # Dodger Blue
    "ENLSP": "00BFFF",  # Deep Sky Blue
    "TACL": "87CEEB",  # Sky Blue
    # CV Conferences (green variants)
    "CVPR": "2E8B57",  # Sea Green
    "ECCV": "3CB371",  # Medium Sea Green
    # Systems Conferences (purple variants)
    "ASPLOS": "9370DB",  # Medium Purple
    "SOSP": "8A2BE2",  # Blue Violet
    "ISCA": "9932CC",  # Dark Orchid
    "MICRO": "BA55D3",  # Medium Orchid
    "MLSys": "DDA0DD",  # Plum
    # Architecture/Hardware (red variants)
    "ATC": "DC143C",  # Crimson
    "DATE": "B22222",  # Fire Brick
    "SC": "CD5C5C",  # Indian Red
    "TC": "F08080",  # Light Coral
    "VLDB": "A52A2A",  # Brown
    "VLSI": "8B0000",  # Dark Red
    # Journals (teal variants)
    "JMLR": "008B8B",  # Dark Cyan
    "TMLR": "20B2AA",  # Light Sea Green
    "Neuromorphic Computing and Engineering": "40E0D0",  # Turquoise
    # Others (neutral colors)
    "arXiv": "1E88E5",  # Bright Blue (Material Design Blue)
    "AutoML Workshop": "A9A9A9",  # Dark Gray
    "Blog": "696969",  # Dim Gray
    "github": "2F4F4F",  # Dark Slate Gray
}


def readMeta():
    pinfos = []
    for year in os.listdir("./meta"):
        if year == "search":
            continue
        for f in os.listdir(f"./meta/{year}"):
            # todo: add year dir
            pinfo = eppb.PaperInfo()
            try:
                with open(os.path.join("./meta", year, f), "r") as rf:
                    pb.text_format.Merge(rf.read(), pinfo)
                pinfos.append((pinfo, f))
            except:
                print("read error in {}".format(f))

    return pinfos


word_pb2str = {
    eppb.Keyword.Word.none: "None",
    eppb.Keyword.Word.sparse_pruning: "Sparse/Pruning",
    eppb.Keyword.Word.quantization: "Quantization",
    eppb.Keyword.Word.survey: "Survey",
    eppb.Keyword.Word.low_rank: "Low Rank Decomposition",
    eppb.Keyword.Word.fusion: "Layer Fusion (Reduce IO)",
    eppb.Keyword.Word.tool: "Tool",
    eppb.Keyword.Word.kv_cache: "KV Cache Optimization/Efficient Attention",
    eppb.Keyword.Word.efficient_training: "Efficient Training",
    eppb.Keyword.Word.structure_design: "Network Structure Design",
    eppb.Keyword.Word.weight_sparsity: "Sparsity (Weight)",
    eppb.Keyword.Word.activation_sparsity: "Sparsity (Activation)",
    eppb.Keyword.Word.attention_sparsity: "Sparsity (Attention)",
    eppb.Keyword.Word.structured_sparsity: "Sparsity (Structured)",
    eppb.Keyword.Word.deployment: "LLM Deployment",
    eppb.Keyword.Word.overlap: "Communication-Computation Overlap",
    eppb.Keyword.Word.modeling: "Modeling"
}

for k, v in word_pb2str.items():
    word_pb2str[k] = f"{k-1:02}-{v}"


def get_table_header():
    return """
| Meta | Title | Cover | Publish | Code | Note |
|:-----|:------|:------|:--------|:-----|:-----|
|<div style="width: 50px"></div>|<div style="width: 200px"></div>|<div style="width: 400px"></div>|<div style="width: 100px"></div>|<div style="width: 100px"></div>|<div style="width: 60px"></div>|
"""


def main():
    args = parse_args()
    columns = [
        "meta",
        "Title",  # (abbr) [title](url)
        "Cover",
        "pub",  # ICLR
        "year",  # 2022
        "codeeeee",  # [type](url)
        "note",  # [](url)
    ]
    pinfos = readMeta()
    data_list = []

    keyword_cls = {}
    year_cls = {}
    pub_cls = {}
    inst_cls = {}
    author_cls = {}

    for pinfo, f in pinfos:
        file_name = f.replace(".prototxt", "")
        year = pinfo.pub.year
        if pinfo.paper.abbr:
            meta = f"[{pinfo.paper.abbr}](./meta/{year}/{f})"
        else:
            meta = f"[m](./meta/{year}/{f})"

        title = ""
        title += pinfo.paper.title
        if pinfo.paper.url:
            title = "[{}]({})".format(title, pinfo.paper.url)

        pub = pinfo.pub.where

        code = ""
        codetype = pinfo.code.type if pinfo.code.type else "code"

        if pinfo.code.url:
            if "github.com" in pinfo.code.url:  # https://github.com/artidoro/qlora
                # ![GitHub Repo stars](https://img.shields.io/github/stars/hustzxd/LSQuantization)
                [user_id, repo] = pinfo.code.url.split("/")[3:5]
                code = f"![GitHub Repo stars](https://img.shields.io/github/stars/{user_id}/{repo})"
            else:
                code = "[{}]({})".format(codetype, pinfo.code.url)

        note = ""
        if pinfo.note.url:
            if os.path.exists(f"./notes/{pinfo.note.url}"):
                note = "[note](./notes/{})".format(pinfo.note.url)
            elif os.path.exists(f"./notes/{year}/{pinfo.note.url}"):
                note = f"[note](./notes/{year}/{pinfo.note.url})"

            elif os.path.exists(f"./notes/{year}/{file_name}/{pinfo.note.url}"):
                note = f"[note](./notes/{year}/{file_name}/{pinfo.note.url})"
            else:
                note = "[note]({})".format(pinfo.note.url)

        cover = ""
        if pinfo.cover.url:
            if os.path.exists(f"./notes/{pinfo.cover.url}"):
                cover = f"./notes/{pinfo.cover.url}"
            elif os.path.exists(f"./notes/{year}/{pinfo.cover.url}"):
                cover = f"./notes/{year}/{pinfo.cover.url}"
            elif os.path.exists(f"./notes/{year}/{file_name}/{pinfo.cover.url}"):
                cover = f"./notes/{year}/{file_name}/{pinfo.cover.url}"
            else:
                cover = pinfo.cover.url

            cover = "<img width='400' alt='image' src='{}'>".format(cover)

        data = [meta, title, cover, pub, year, code, note]

        if pinfo.pub.year:
            if pinfo.pub.year in year_cls:
                year_cls[pinfo.pub.year].append(data)
            else:
                year_cls[pinfo.pub.year] = [data]

        if pinfo.pub.where:
            pub_ = pinfo.pub.where
            if pub_ in pub_cls:
                pub_cls[pub_].append(data)
            else:
                pub_cls[pub_] = [data]

        if pinfo.paper.institutions:
            for inst in pinfo.paper.institutions:
                if inst in inst_cls:
                    inst_cls[inst].append(data)
                else:
                    inst_cls[inst] = [data]

        if pinfo.paper.authors:
            for authors in pinfo.paper.authors:
                authors = authors.split(",")
                for author in authors:
                    author = author.strip()
                    if author in author_cls:
                        author_cls[author].append(data)
                    else:
                        author_cls[author] = [data]

        if pinfo.keyword.words:
            for word in pinfo.keyword.words:
                word = word_pb2str[word]
                if word in keyword_cls:
                    keyword_cls[word].append(data)
                else:
                    keyword_cls[word] = [data]

        data_list.append(data)

    df = pd.DataFrame(data_list, columns=columns)
    df = df.sort_values(by=["year", "pub", "Title"], ascending=True).reset_index(drop=True)
    with open("README_prefix.md") as rf:
        markdown = rf.read()
    markdown += "\n\n"

    markdown += "\n## Paper List\n\n"

    markdown += gen_list(year_cls, columns, "year", is_open=True, reverse=True)

    with open("README_suffix.md") as rf:
        markdown += rf.read()

    with open("README.md", "w") as wf:
        wf.write(markdown)
    print("Generate README.md done")

    del_keys = []
    for k, v in author_cls.items():
        if len(v) == 1:
            del_keys.append(k)
    for k in del_keys:
        author_cls.pop(k)

    cls_dict = {
        "keyword": keyword_cls,
        "year": year_cls,
        "publication": pub_cls,
        "institution": inst_cls,
        "author": author_cls,
    }
    for cls_name in ["keyword", "year", "publication", "institution", "author"]:
        with open(f"cls_{cls_name}.md", "w") as wf:
            wf.write(gen_table(cls_dict[cls_name], columns, cls_name, is_open=True, reverse=(cls_name == "year")))


def colorful_text(text, year=None, color="green"):
    if text in PUBLISH_COLOR:
        color = PUBLISH_COLOR[text]
    if year is not None:
        text = f"{year}-{text}"
    text = text.replace(" ", "_")
    template = "![Publish](https://img.shields.io/badge/{}-{})"
    return template.format(text, color)


def gen_table(out_cls, columns, cls_name, is_open=False, reverse=False):
    markdown = ""
    out_cls = dict(sorted(out_cls.items(), reverse=reverse))
    if is_open:
        markdown += """<details open><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    else:
        markdown += """<details><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    for key, data in out_cls.items():
        data_ = copy.deepcopy(data)
        for d in data_:
            d[3] = colorful_text(d[3], d[4])
            d.pop(4)
        columns_ = copy.deepcopy(columns)
        columns_[3] = "Publish"
        columns_.pop(4)
        df_ = pd.DataFrame(data_, columns=columns_)
        df_ = df_.sort_values(by=["Publish", "Title"], ascending=True).reset_index(drop=True)
        if is_open:
            markdown += """<details open><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        else:
            markdown += """<details><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        
        # 使用自定义表格头
        markdown += get_table_header()
        
        # 直接使用DataFrame的值，跳过表头
        for row in df_.values:
            # 确保每行有6列数据
            row_data = list(row)
            markdown += "| {} | {} | {} | {} | {} | {} |\n".format(*row_data)
        markdown += "</p>\n</details>\n"
    markdown += "</p>\n</details>\n\n"
    return markdown


def gen_list(out_cls, columns, cls_name, is_open=False, reverse=False):
    markdown = ""
    out_cls = dict(sorted(out_cls.items(), reverse=reverse))
    if is_open:
        markdown += """<details open><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    else:
        markdown += """<details><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    for key, data in out_cls.items():
        df_ = pd.DataFrame(data, columns=columns)
        df_ = df_.sort_values(by=["year", "pub", "Title"], ascending=True).reset_index(drop=True)
        if is_open:
            markdown += """<details open><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        else:
            markdown += """<details><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        # markdown += df_.to_markdown()
        for index, row in df_.iterrows():
            line_ = f"{index+1}. {row["Title"]} [{colorful_text(row["pub"], row["year"])}] {row["codeeeee"]} \n"
            markdown += line_
        markdown += "</p>\n</details>\n"
    markdown += "</p>\n</details>\n\n"
    return markdown


if __name__ == "__main__":
    main()
