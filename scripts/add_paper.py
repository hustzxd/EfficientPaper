import argparse
import datetime
import hashlib
import os
import random
import re
import string
import sys

import arxiv
import google.protobuf as pb
import google.protobuf.text_format

# Add project root to path before importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import efficient_paper_pb2 as eppb

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper INFO")
    parser.add_argument("-f", "--file", default=None, help="The file name")
    args = parser.parse_args()
    return args


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def extract_code_url(summary):
    """Extract GitHub/code repository URL from paper summary.

    Args:
        summary: Paper abstract/summary text

    Returns:
        str: Code repository URL if found, None otherwise
    """
    if not summary:
        return None

    # Common patterns for code repositories
    patterns = [
        r'https?://github\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://huggingface\.co/[\w\-\.]+/[\w\-\.]+',
        r'https?://gitee\.com/[\w\-\.]+/[\w\-\.]+',
    ]

    for pattern in patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            url = match.group(0)
            # Remove trailing punctuation
            url = re.sub(r'[.,;:\)]+$', '', url)
            return url

    return None


def extract_institutions(authors):
    """Extract institutions from arXiv author objects.

    Args:
        authors: List of arxiv.Result.Author objects

    Returns:
        list: List of unique institution names
    """
    institutions = []
    seen = set()

    for author in authors:
        # Try to get affiliation from author object
        if hasattr(author, 'affiliation') and author.affiliation:
            affiliation = author.affiliation.strip()
            if affiliation and affiliation not in seen:
                institutions.append(affiliation)
                seen.add(affiliation)

    return institutions


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
        if args.file is not None and os.path.exists(args.file):
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

        # Automatically extract institutions
        institutions = extract_institutions(paper.authors)
        if institutions:
            print(f"\033[92m✓ Found {len(institutions)} institution(s):\033[0m")
            for inst in institutions:
                print(f"  - {inst}")
            try:
                pinfo.paper.institutions.clear()
            except:
                pass
            pinfo.paper.institutions.extend(institutions)
        else:
            print("\033[93m⚠ No institutions found in arXiv metadata\033[0m")

        # Automatically extract code repository URL
        code_url = extract_code_url(paper.summary)
        if code_url:
            print(f"\033[92m✓ Found code repository: {code_url}\033[0m")
            pinfo.code.url = code_url
        else:
            print("\033[93m⚠ No code repository URL found in abstract\033[0m")
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
    # Ensure year-based directory exists
    year_dir = os.path.join(root_dir, "meta", str(pinfo.pub.year))
    os.makedirs(year_dir, exist_ok=True)

    # Check if file already exists
    new_path = os.path.join(year_dir, f"{name}.prototxt")
    old_path = os.path.join(root_dir, "meta", f"{name}.prototxt")

    if os.path.exists(new_path) or os.path.exists(old_path):
        print("The file `{}` already exists, please use another name".format(name))
        return

    with open(new_path, "w") as wf:
        print(pinfo)
        print("Writing paper information into {}/meta/{}/{}.prototxt".format(root_dir, pinfo.pub.year, name))
        wf.write(str(pinfo))

    # mkdir note with year-based structure
    note_dir = f"notes/{pinfo.pub.year}/{name}"
    os.makedirs(note_dir, exist_ok=True)
    note_file = os.path.join(note_dir, "note.md")

    if not os.path.exists(note_file):
        if paper is not None:
            title = paper.title
            summary = paper.summary
        else:
            title = name
            summary = ""
        note_content = f"""# {title}\n\n"""
        if paper is not None:
            note_content += f"""> {", ".join(authors)}\n\n"""
        # note_content += """<p align="center">\n<img src="../../blank.jpg" width="600" title="blank">\n</p>\n\n"""
        note_content += f"""![111](../../blank.jpg)\n\n"""
        note_content += f"""## Abstract\n\n{summary}\n"""
        with open(note_file, "w") as wf:
            wf.write(note_content)


if __name__ == "__main__":
    main()
