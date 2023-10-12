import os
import shutil
import sys

import google.protobuf as pb
import google.protobuf.text_format
import ipdb
import pandas as pd

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")


def readMeta():
    pinfos = []
    for f in os.listdir("./meta"):
        pinfo = eppb.PaperInfo()
        try:
            with open(os.path.join("./meta", f), "r") as rf:
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
    eppb.Keyword.Word.working: "0 Working"
}

TITLE = "ttttttttttttttttttttttttttttttitle"
COVER = "ccccccccccccccccccover"

def main():
    columns = [
        "meta",
        TITLE,  # (abbr) [title](url)
        COVER,
        "pub",  # ICLR
        "year",  # 2022
        "code",  # [type](url)
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
        if pinfo.paper.abbr:
            meta = "[{}](./meta/{})".format(pinfo.paper.abbr, f)
        else:
            meta = "[m](./meta/{})".format(f)

        title = ""
        title += pinfo.paper.title
        if pinfo.paper.url:
            title = "[{}]({})".format(title, pinfo.paper.url)

        pub = pinfo.pub.where
        year = pinfo.pub.year

        code = ""
        codetype = pinfo.code.type if pinfo.code.type else "code"

        if pinfo.code.url:
            code = "[{}]({})".format(codetype, pinfo.code.url)

        note = ""
        if pinfo.note.url:
            if pinfo.note.url.endswith(".md"):
                note = "[note](./notes/{})".format(pinfo.note.url)
            else:
                note = "[note]({})".format(pinfo.note.url)

        cover = ""
        if pinfo.cover.url:
            if not pinfo.cover.url.startswith("http"):
                cover = "./notes/{}".format(pinfo.cover.url)
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
            for author in pinfo.paper.authors:
                # author = author.replace(" ", "-")
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

    # keyword_cls = {}
    # year_cls = {}
    # pub_cls = {}
    # inst_cls = {}
    # author_cls = {}

    markdown += gen_table(keyword_cls, columns, "keyword", is_open=True)
    markdown += gen_table(year_cls, columns, "year")
    markdown += gen_table(pub_cls, columns, "publication")
    markdown += gen_table(inst_cls, columns, "instution")
    markdown += gen_table(author_cls, columns, "author")

    with open("README_suffix.md") as rf:
        markdown += rf.read()

    with open("README.md", "w") as wf:
        wf.write(markdown)
    print("Generate README.md done")


def gen_table(out_cls, columns, cls_name, is_open=False):
    markdown = ""
    out_cls = dict(sorted(out_cls.items(), reverse=False))
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
