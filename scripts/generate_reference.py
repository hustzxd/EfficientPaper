import sys
import os
import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")


def readMeta():
    repo_list = eppb.RepoList()
    try:
        with open(os.path.join("./meta", "search", "references.prototxt"), "r") as rf:
            pb.text_format.Merge(rf.read(), repo_list)
    except:
        print("read error in references.prototxt")

    return repo_list


if __name__ == "__main__":
    repo_list = readMeta()
    with open("./README_suffix.md", "w") as wf:
        wf.write("## References\n\n")
        for i, url in enumerate(repo_list.repo_url):
            if "github.com" in url:  # https://github.com/artidoro/qlora
                [user_id, repo] = url.split("/")[3:5]
                code = f"[{url}]({url}) [[![GitHub Repo stars](https://img.shields.io/github/stars/{user_id}/{repo})](https://github.com/{user_id}/{repo})]"
            else:
                code = "[{}]({})".format(url, url)
            wf.write(f"{i+1}. {code}\n\n")
    print("generate_reference.py done")