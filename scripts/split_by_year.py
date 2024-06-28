import os
import shutil
import sys

import google.protobuf as pb
import google.protobuf.text_format
import ipdb

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")

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
