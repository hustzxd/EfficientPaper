import argparse
import hashlib
import os
import random
import shutil
import string
import sys

import google.protobuf as pb
import google.protobuf.text_format
import ipdb

sys.path.append("./")

from proto import efficient_paper_pb2 as eppb


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def main():
    parser = argparse.ArgumentParser(description="Paper Info")
    parser.add_argument("--name", type=str, help="Please add short name for a paper")
    parser.add_argument("--note", action="store_true", help="Whether to setup the note directory.")
    args = parser.parse_args()

    pinfo = eppb.PaperInfo()
    with open("proto/template.prototxt", "r") as rf:
        pb.text_format.Merge(rf.read(), pinfo)

    if args.name:
        name = args.name
    else:
        name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    # root_dir = os.getenv("CURRENT_DIR")
    root_dir = "./"
    if os.path.exists(os.path.join(root_dir, "meta", "{}.prototxt".format(name))):
        print("The file `{}` already exists, please use another name".format(name))
        return

    with open(os.path.join(root_dir, "meta", "{}.prototxt".format(name)), "w") as wf:
        print(pinfo)
        print("Writing paper information into {}/meta/{}.prototxt".format(root_dir, name))
        wf.write(str(pinfo))
    # mkdir note
    if args.note:
        os.makedirs(f"notes/{name}", exist_ok=True)
        if not os.path.exists(f"notes/{name}/note.md"):
            note_content = (
                f"""# {name}\n\n<p align="center">\n<img src="../blank.jpg" width="600" title="blank">\n</p>\n"""
            )
            with open(f"notes/{name}/note.md", "w") as wf:
                wf.write(note_content)


if __name__ == "__main__":
    main()
