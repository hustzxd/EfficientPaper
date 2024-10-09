import datetime as DT
import os

import arxiv


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

    key_words = ["sparse", "pruning", "sparsity", "quantize", "quantization", "low-bit"]

    query = ""
    for i, k in enumerate(key_words):
        if i == 0:
            query = f"abs:{k}"
        query += f" OR abs:{k}"
    query = f"({query}) AND (abs:LLM OR abs:LLMs OR abs:attention OR abs:transformer) ANDNOT (abs:spiking)"
    query = query.replace("(", "%28")
    query = query.replace(")", "%29")
    print(query)
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