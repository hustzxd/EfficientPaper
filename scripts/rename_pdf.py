import argparse
import os
import string

from pdftitle import get_title_from_file

CNT_FAILED = 0
CNT_SUCCESS = 0


def rename_all_files(rootdir):
    global CNT_FAILED
    global CNT_SUCCESS
    l1 = os.listdir(rootdir)
    for i in range(0, len(l1)):
        path = os.path.join(rootdir, l1[i])
        if os.path.isdir(path):
            rename_all_files(path)
        if os.path.isfile(path) and path.endswith(".pdf"):
            try:
                title = get_title_from_file(path)
                new_name = title.lower()  # Lower case name
                valid_chars = set(string.ascii_lowercase + string.digits + " ")
                new_name = "".join(c for c in new_name if c in valid_chars)
                new_name = new_name.replace(" ", "_") + ".pdf"
                if is_better_name(l1[i], new_name):
                    new_name_path = os.path.join(rootdir, new_name)
                    os.rename(path, new_name_path)
                    print("{} => {}".format(path, new_name))
                    CNT_SUCCESS += 1
                CNT_FAILED += 1
            except Exception:
                CNT_FAILED += 1
                pass


def is_better_name(old_name, new_name):
    if "_" not in new_name:
        return False
    if len(old_name) >= len(new_name):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pdftitle", description="Extracts the title of a PDF article", epilog=""
    )
    parser.add_argument("--dir", help="pdf directory", default="pdfs", required=False)
    args = parser.parse_args()
    rename_all_files(args.dir)
    print("Rename Successful/Failed : {}/{}".format(CNT_SUCCESS, CNT_FAILED))
