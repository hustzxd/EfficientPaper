import argparse
import os
import string

from pdftitle import GetTitleParameters, get_title_from_file

CNT_FAILED = 0
CNT_SUCCESS = 0
CNT_SKIPPED = 0


def rename_all_files(rootdir):
    global CNT_FAILED
    global CNT_SUCCESS
    global CNT_SKIPPED
    l1 = os.listdir(rootdir)
    get_title_param = GetTitleParameters(replace_missing_char="_")
    for i in range(0, len(l1)):
        path = os.path.join(rootdir, l1[i])
        if os.path.isdir(path):
            rename_all_files(path)
        if os.path.isfile(path) and path.endswith(".pdf"):
            try:
                title = get_title_from_file(path, get_title_param)
                if title is None:
                    print(f"Failed to extract title from {path}")
                    CNT_FAILED += 1
                    continue
                    
                new_name = title.lower()  # Lower case name
                valid_chars = set(string.ascii_lowercase + string.digits + " ")
                new_name = "".join(c for c in new_name if c in valid_chars)
                new_name = new_name.replace(" ", "_") + ".pdf"
                
                if is_better_name(l1[i], new_name):
                    new_name_path = os.path.join(rootdir, new_name)
                    os.rename(path, new_name_path)
                    print("{} => {}".format(path, new_name))
                    CNT_SUCCESS += 1
                else:
                    print(f"Skipped {path} - new name not better")
                    CNT_SKIPPED += 1
            except Exception as e:
                CNT_FAILED += 1
                print(f"Error processing {path}: {e}")


def is_better_name(old_name, new_name):
    # 新名字必须包含下划线（表示有空格被替换）
    if "_" not in new_name:
        return False
    # 新名字不能太短（避免提取标题失败的情况）
    if len(new_name) < 10:
        return False
    # 新名字不应该比旧名字短（避免信息丢失）
    if len(old_name) >= len(new_name):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pdftitle", description="Extracts the title of a PDF article", epilog="")
    parser.add_argument("--dir", help="pdf directory", default="pdfs", required=False)
    args = parser.parse_args()
    rename_all_files(args.dir)
    print("\nSummary:")
    print("Rename Successful: {}".format(CNT_SUCCESS))
    print("Rename Failed: {}".format(CNT_FAILED))
    print("Rename Skipped: {}".format(CNT_SKIPPED))
