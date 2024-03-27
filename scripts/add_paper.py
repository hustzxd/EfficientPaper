import argparse
import hashlib
import os
import random
import shutil
import string
import sys
import arxiv

import google.protobuf as pb
import google.protobuf.text_format
import ipdb

sys.path.append("./")

from proto import efficient_paper_pb2 as eppb


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def main():

    name = input("Please add short name for a paper (Default: random code) \n abbr name: ")
    if len(name) == 0:
        name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    print("Paper name is: " + name)

    arxiv_id = input("Please input the abs of arXiv (Default: None) \n arxiv_id: ")
    if len(arxiv_id) != 0:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    else:
        paper = None

    pinfo = eppb.PaperInfo()
    with open("proto/template.prototxt", "r") as rf:
        pb.text_format.Merge(rf.read(), pinfo)

    if paper is not None:
        pinfo.paper.title = paper.title
        pinfo.paper.url = paper.entry_id
        authors = [author.name for author in paper.authors]
        pinfo.paper.authors.clear()
        pinfo.paper.authors.extend(authors)
        pinfo.pub.year = paper.published.year

    root_dir = "./"
    if os.path.exists(os.path.join(root_dir, "meta", "{}.prototxt".format(name))):
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
        else:
            title = name
        note_content = (
            f"""# {title}\n\n<p align="center">\n<img src="../../blank.jpg" width="600" title="blank">\n</p>\n\n## Abstract\n\n{paper.summary}\n"""
        )
        with open(f"notes/{name}/note.md", "w") as wf:
            wf.write(note_content)


if __name__ == "__main__":
    main()
