import datetime as DT
import os
import sys
import time

import arxiv
import google.protobuf as pb
import google.protobuf.text_format

sys.path.insert(0, "/Users/xiandong/projects/EfficientPaper")

from proto import efficient_paper_pb2 as eppb


def fetch_papers_with_retry(client, search, max_retries=5):
    """Fetch papers with exponential backoff retry on rate limit errors."""
    retry_count = 0
    base_delay = 10  # Start with 10 seconds

    while retry_count < max_retries:
        try:
            papers = []
            for paper in client.results(search):
                papers.append(paper)
                # Add a small delay between individual paper fetches
                time.sleep(0.5)
            return papers
        except arxiv.HTTPError as e:
            if "429" in str(e):
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\nMax retries ({max_retries}) reached. Giving up.")
                    raise

                delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                print(f"\nRate limit hit (429 error). Retrying in {delay} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            raise

    return []


def main():
    today = DT.date.today()
    dir_root = "/Users/xiandong/projects/EfficientPaper/weekly_paper"
    files = [f for f in os.listdir(dir_root) if f.endswith('.md')]
    files.sort()
    previous_day = files[-1].replace(".md", "")
    previous_day = DT.date.fromisoformat(previous_day)
    print(f"Previous update date: {previous_day}")
    if today == previous_day:
        print("Already up to date")
        return

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

    # Search for papers in batches
    total_results = 300  # 设置想要获取的总论文数
    papers_per_batch = 100  # arXiv API的每页限制
    markdown_content = f"# {today}\n\n"
    papers_found = 0

    # First search to get all results
    search = arxiv.Search(
        query=query,
        max_results=total_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    try:
        # Fetch all papers with retry logic
        print("Fetching papers from arXiv...")
        papers = fetch_papers_with_retry(client, search)
        print(f"Successfully fetched {len(papers)} papers")

        # Process all papers
        for i, paper in enumerate(papers, 1):
            date = paper.published.date()
            if date >= previous_day:
                title = paper.title
                authors = [author.name for author in paper.authors]
                authors = ", ".join(authors)
                url = paper.entry_id
                summary = paper.summary
                key_words_has = False
                bg_words_has = False
                for k in key_words:
                    if k in summary:
                        summary = summary.replace(k, f"![key](https://img.shields.io/badge/{k}-F08080)")
                        key_words_has = True
                for k in bg_words:
                    if k in summary:
                        summary = summary.replace(k, f"![key](https://img.shields.io/badge/{k}-FF8C00)")
                        bg_words_has = True
                if key_words_has and bg_words_has:
                    print(title)
                    markdown_content += f"## {title}\n\n".replace(":", "")
                    markdown_content += f">Authors: {authors}\n\n"
                    markdown_content += f">{date}\n\n"
                    markdown_content += f"> [{url}]({url})\n\n"
                    markdown_content += f"{summary}\n\n\n"
                    papers_found += 1

            # Progress update
            if i % 50 == 0:
                print(f"Processed {i}/{len(papers)} papers, found {papers_found} matching papers so far...")

    except arxiv.UnexpectedEmptyPageError:
        print(f"\nReached end of results at {papers_found} papers")
    except Exception as e:
        print(f"\nError occurred: {e}")

    print(f"\nTotal papers found: {papers_found}")
    
    if papers_found > 0:
        file_name = f"/Users/xiandong/projects/EfficientPaper/weekly_paper/{today}.md"
        with open(file_name, "w") as wf:
            wf.write(markdown_content)
        
        os.system(f"/Users/xiandong/miniconda3/bin/markdown-toc {file_name}")
        os.system(f"/Users/xiandong/miniconda3/bin/python /Users/xiandong/projects/EfficientPaper/scripts/fix_mkdocs_links.py {file_name}")

    else:
        print("No new papers found, skipping file creation")


if __name__ == "__main__":
    main()
