import argparse
import os
import shutil
import sys

import google.protobuf as pb
import google.protobuf.text_format
import ipdb
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


def readMeta():
    pinfos = []
    for year in os.listdir("./meta"):
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
    eppb.Keyword.Word.cache: "Cache Optimization",
    eppb.Keyword.Word.working: "0 Working",
    eppb.Keyword.Word.training: "Efficient Training",
    eppb.Keyword.Word.structure_design: "Network Structure Design",
}

TITLE = "ttttttttttttttttttttttttttttttitle"
COVER = "ccccccccccccccccccover"


def main():
    args = parse_args()
    columns = [
        "meta",
        TITLE,  # (abbr) [title](url)
        COVER,
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

        # data = [meta, title, pub, year, code, note, cover]
        data = [meta, title, cover, pub, year, code, note]

        if pinfo.pub.year:
            if pinfo.pub.year in year_cls:
                year_cls[pinfo.pub.year].append(data)
            else:
                year_cls[pinfo.pub.year] = [data]

        if pinfo.pub.where:
            # pub_ = pinfo.pub.where.replace(" ", "-")
            pub_ = pinfo.pub.where
            if pub_ in pub_cls:
                pub_cls[pub_].append(data)
            else:
                pub_cls[pub_] = [data]

        if pinfo.paper.institutions:
            for inst in pinfo.paper.institutions:
                # inst = inst.replace(" ", "-")
                if inst in inst_cls:
                    inst_cls[inst].append(data)
                else:
                    inst_cls[inst] = [data]

        if pinfo.paper.authors:
            for authors in pinfo.paper.authors:
                # author = author.replace(" ", "-")
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
                # word = word.replace(" ", "-")
                if word in keyword_cls:
                    keyword_cls[word].append(data)
                else:
                    keyword_cls[word] = [data]

        data_list.append(data)

    df = pd.DataFrame(data_list, columns=columns)
    df = df.sort_values(by=["year", "pub", TITLE], ascending=True).reset_index(drop=True)
    with open("README_prefix.md") as rf:
        markdown = rf.read()
    markdown += "\n\n"

    markdown += "\n## Paper List\n\n"

    markdown += gen_table(keyword_cls, columns, "keyword", is_open=True)

    # if args.detail:
    #     markdown += gen_table(author_cls, columns, "author")

    with open("README_suffix.md") as rf:
        markdown += rf.read()

    with open("README.md", "w") as wf:
        wf.write(markdown)
    print("Generate README.md done")
    cls_dict = {"year": year_cls, "publication": pub_cls, "institution": inst_cls, "author": author_cls}
    for cls_name in ["year", "publication", "institution", "author"]:
        with open(f"cls_{cls_name}.md", "w") as wf:
            wf.write(gen_table(cls_dict[cls_name], columns, cls_name, is_open=True, reverse=(cls_name == "year")))


def gen_table(out_cls, columns, cls_name, is_open=False, reverse=False):
    markdown = ""
    out_cls = dict(sorted(out_cls.items(), reverse=reverse))
    if is_open:
        markdown += """<details open><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    else:
        markdown += """<details><summary>\n\n### {}\n</summary> \n<p>\n\n""".format(cls_name)
    for key, data in out_cls.items():
        df_ = pd.DataFrame(data, columns=columns)
        df_ = df_.sort_values(by=["year", "pub", TITLE], ascending=True).reset_index(drop=True)
        if is_open:
            markdown += """<details open><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        else:
            markdown += """<details><summary><b>{}</b></summary> \n<p>\n\n""".format(key)
        markdown += df_.to_markdown()
        markdown += "</p>\n</details>\n"
    markdown += "</p>\n</details>\n\n"
    return markdown


if __name__ == "__main__":
    main()
