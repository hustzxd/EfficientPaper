import argparse
import datetime
import hashlib
import os
import random
import string
import sys

import arxiv
import google.protobuf as pb
import google.protobuf.text_format

sys.path.append("./")

from proto import efficient_paper_pb2 as eppb


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper INFO")
    parser.add_argument("-f", "--file", default=None, help="The file name")
    args = parser.parse_args()
    return args


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


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
        if os.path.exists(args.file):
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
        authors = [author.name for author in paper.authors]
        try:
            pinfo.paper.authors.clear()
        except:
            pinfo.paper.authors.pop()
            pinfo.paper.authors.pop()
        pinfo.paper.authors.extend(authors)
        pinfo.pub.year = paper.published.year
    else:
        current_year = datetime.date.today().year
        year = input(f"Please input the public year of this paper (Default: {current_year}) \n Year: ")
        if len(year) != 0:
            assert 1990 <= int(year) <= 2030
            year = int(year)
        else:
            year = 2025
        pinfo.pub.year = year

    root_dir = "./"
    if os.path.exists(os.path.join(root_dir, "meta", f"{pinfo.pub.year}", f"{name}.prototxt")) or os.path.exists(
        os.path.join(root_dir, "meta", f"{name}.prototxt")
    ):
        print("The file `{}` already exists, please use another name".format(name))
        return

    with open(os.path.join(root_dir, "meta", "{}.prototxt".format(name)), "w") as wf:
        print(pinfo)
        print("Writing paper information into {}/meta/{}.prototxt".format(root_dir, name))
        wf.write(str(pinfo))
    # mkdir note
    os.makedirs(f"notes/{name}", exist_ok=True)
    if not os.path.exists(f"notes/{name}/note.md"):
        if paper is not None:
            title = paper.title
            summary = paper.summary
        else:
            title = name
            summary = ""
        note_content = f"""# {title}\n\n"""
        if paper is not None:
            note_content += f"""> {", ".join(authors)}\n\n"""
        note_content += """<p align="center">\n<img src="../../blank.jpg" width="600" title="blank">\n</p>\n\n"""
        note_content += f"""## Abstract\n\n{summary}\n"""
        with open(f"notes/{name}/note.md", "w") as wf:
            wf.write(note_content)


if __name__ == "__main__":
    main()
