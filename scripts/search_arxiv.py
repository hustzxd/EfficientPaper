import datetime as DT
import os
import sys

import arxiv
import google.protobuf as pb
import google.protobuf.text_format

sys.path.append("/Users/xiandong/projects/EfficientPaper")

from proto import efficient_paper_pb2 as eppb


def main():
    today = DT.date.today()
    dir_root = "/Users/xiandong/projects/EfficientPaper/weekly_paper"
    files = os.listdir(dir_root)
    files.sort()
    previous_day = files[-1].replace(".md", "")
    previous_day = DT.date.fromisoformat(previous_day)
    print(f"Previou update date: {previous_day}")
    if today == previous_day:
        print("Already up to date")
        return
    # week_ago = today - DT.timedelta(days=7)

    search_words = eppb.SearchWord()
    try:
        with open(
            os.path.join("/Users/xiandong/projects/EfficientPaper/meta", "search", "efficient_keywords.prototxt"), "r"
        ) as rf:
            pb.text_format.Merge(rf.read(), search_words)
    except:
        print("read error")

    bg_words = search_words.background_words
    key_words = search_words.key_words
    exclude_words = search_words.exclude_words

    bg_query = ""
    for i, k in enumerate(bg_words):
        if i == 0:
            bg_query = f"abs:{k}"
        bg_query += f" OR abs:{k}"

    key_query = ""
    for i, k in enumerate(key_words):
        if i == 0:
            key_query = f"abs:{k}"
        key_query += f" OR abs:{k}"

    exclude_query = ""
    for i, k in enumerate(exclude_words):
        if i == 0:
            exclude_query = f"abs:{k}"
        exclude_query += f" OR abs:{k}"

    query = f"({key_query}) AND ({bg_query}) ANDNOT ({exclude_query})"

    print(query)
    query = query.replace("(", "%28")
    query = query.replace(")", "%29")

    # Construct the default API client.
    client = arxiv.Client()

    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
        query=query,
        max_results=400,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    markdown_content = f"# {today}\n\n"
    # `results` is a generator; you can iterate over its elements one by one...
    for paper in client.results(search):
        date = paper.published.date()
        if date >= previous_day:
            title = paper.title
            print(title)
            authors = [author.name for author in paper.authors]
            authors = ", ".join(authors)
            url = paper.entry_id
            summary = paper.summary
            for k in key_words:
                if k in summary:
                    summary = summary.replace(k, f"**{k}**")
            markdown_content += f"## {title}\n\n".replace(":", "")
            markdown_content += f">Authors: {authors}\n\n"
            markdown_content += f">{date}\n\n"
            markdown_content += f"> {url}\n\n"
            markdown_content += f"{summary}\n\n\n"

    file_name = f"/Users/xiandong/projects/EfficientPaper/weekly_paper/{today}.md"
    with open(file_name, "w") as wf:
        wf.write(markdown_content)

    os.system(f"/Users/xiandong/miniconda3/bin/markdown-toc {file_name}")


if __name__ == "__main__":
    main()
