import os

import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_paper_pb2 as eppb


def main():
    gene_base_template()


def gene_base_template():
    # default values
    pinfo = eppb.PaperInfo()
    pinfo.paper.title = pinfo.paper.title
    pinfo.paper.abbr = pinfo.paper.abbr
    pinfo.paper.url = pinfo.paper.url
    pinfo.paper.authors.extend(["Name1", "Name2"])
    pinfo.paper.institutions.extend(["inst1", "inst2"])

    pinfo.pub.where = pinfo.pub.where
    pinfo.pub.year = pinfo.pub.year

    pinfo.code.type = pinfo.code.type
    pinfo.code.url = pinfo.code.url

    pinfo.note.url = pinfo.note.url

    pinfo.cover.url = pinfo.cover.url

    pinfo.baseline.methods.extend(["None"])

    pinfo.keyword.words.extend([pinfo.keyword.Word.none])

    root_dir = os.getenv("CURRENT_DIR")
    with open(os.path.join(root_dir, "proto", "template.prototxt"), "w") as wf:
        print(pinfo)
        print("Writing paper information into {}/proto/template.prototxt".format(root_dir))
        wf.write(str(pinfo))


if __name__ == "__main__":
    main()
