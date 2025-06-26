import os
import shutil
import sys

import google.protobuf as pb
import google.protobuf.text_format
import ipdb

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")


def init_year_dir():
    for f in os.listdir("./meta"):
        pinfo = eppb.PaperInfo()
        if f.endswith(".prototxt"):
            try:
                with open(os.path.join("./meta", f), "r") as rf:
                    pb.text_format.Merge(rf.read(), pinfo)
            except:
                print("read error in {}".format(f))
            year = pinfo.pub.year
            if year == 0:
                continue
            os.makedirs(f"./meta/{year}", exist_ok=True)
            print(f"Move ./meta/{f} to ./meta/{year}/{f}")
            if os.path.exists(f"./meta/{year}/{f}"):
                print(f"./meta/{year}/{f} already exists, Please rename ./meta/{f}")
            else:
                shutil.move(f"./meta/{f}", f"./meta/{year}/{f}")

                note_name = f.replace(".prototxt", "")
                if os.path.exists(f"./notes/{note_name}"):
                    os.makedirs(f"./notes/{year}", exist_ok=True)
                    print(f"Move ./notes/{note_name} to ./notes/{year}/{note_name}")
                    shutil.move(f"./notes/{note_name}", f"./notes/{year}/{note_name}")


def check_year_dir():
    pinfos = []
    for year_dir in os.listdir("./meta"):
        if year_dir == "search":
            continue
        for f in os.listdir(f"./meta/{year_dir}"):
            # todo: add year dir
            pinfo = eppb.PaperInfo()
            try:
                with open(os.path.join("./meta", year_dir, f), "r") as rf:
                    pb.text_format.Merge(rf.read(), pinfo)
                pinfos.append((pinfo, f))
            except:
                print("read error in {}".format(f))
            year_info = pinfo.pub.year
            if year_dir == 0:
                continue
            if year_info == int(year_dir):
                continue
            os.makedirs(f"./meta/{year_info}", exist_ok=True)
            print(f"Move ./meta/{year_dir}/{f} to ./meta/{year_info}/{f}")
            if os.path.exists(f"./meta/{year_info}/{f}"):
                print(f"./meta/{year_info}/{f} already exists, Please rename ./meta/{year_dir}/{f}")
            else:
                shutil.move(f"./meta/{year_dir}/{f}", f"./meta/{year_info}/{f}")

                note_name = f.replace(".prototxt", "")
                if os.path.exists(f"./notes/{year_dir}/{note_name}"):
                    os.makedirs(f"./notes/{year_info}", exist_ok=True)
                    print(f"Move ./notes/{year_dir}/{note_name} to ./notes/{year_info}/{note_name}")
                    shutil.move(f"./notes/{year_dir}/{note_name}", f"./notes/{year_info}/{note_name}")



if __name__ == "__main__":
    init_year_dir()
    check_year_dir()