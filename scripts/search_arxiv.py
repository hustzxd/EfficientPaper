import datetime as DT

import arxiv

today = DT.date.today()
week_ago = today - DT.timedelta(days=7)

key_words = ["sparse", "pruning", "sparsity", "quantize", "quantization", "low-bit"]

query = ""
for i, k in enumerate(key_words):
    if i == 0:
        query = f"abs:{k}"
    query += f" OR abs:{k}"
query = f"({query}) AND (LLM OR attention)"
query = query.replace("(", "%28")
query = query.replace(")", "%29")
print(query)
# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
    query=query,
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

markdown_content = ""
# `results` is a generator; you can iterate over its elements one by one...
for paper in client.results(search):
    date = paper.published.date()
    if date > week_ago:
        title = paper.title
        print(title)
        authors = [author.name for author in paper.authors]
        authors = ",".join(authors)
        url = paper.entry_id
        summary = paper.summary
        markdown_content += f"# {title}\n\n"
        markdown_content += f">Authors: {authors}\n\n"
        markdown_content += f">{date}\n\n"
        markdown_content += f"> {url}\n\n"
        markdown_content += f"{summary}\n\n\n"

with open(f"/Users/xiandong/projects/EfficientPaper/weekly_paper/{today}.md", "w") as wf:
    wf.write(markdown_content)
