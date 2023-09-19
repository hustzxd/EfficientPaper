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
import pandas as pd

sys.path.append("./")

from proto import efficient_paper_pb2 as eppb


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def main():
    parser = argparse.ArgumentParser(description="Paper Info")
    parser.add_argument("--name", type=str, help="Please add short name for a paper")

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


if __name__ == "__main__":
    main()
