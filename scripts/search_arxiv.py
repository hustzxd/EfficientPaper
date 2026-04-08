import datetime as DT
import json
import os
import shutil
import sys
import time

import arxiv
from openai import OpenAI
import google.protobuf as pb
import google.protobuf.text_format

sys.path.insert(0, "/Users/xiandong/projects/EfficientPaper")

from proto import efficient_paper_pb2 as eppb


RELEVANCE_PROMPT = """You are a paper relevance filter for LLM efficiency/optimization research.

Given a list of paper titles, return ONLY the indices of papers that are relevant to:
- LLM/Transformer inference optimization (quantization, pruning, sparsity, KV cache, speculative decoding, etc.)
- LLM serving systems (batching, scheduling, disaggregation, offloading, parallelism, etc.)
- Model compression (distillation, low-rank, low-bit, etc.)
- LLM training efficiency (communication, pipeline parallelism, etc.)
- Hardware-aware LLM optimization (GPU, accelerator)

Exclude papers about:
- Vision-only models (image/video generation, detection, segmentation) unless tied to LLM efficiency
- Robotics, medical, biology, physics, math theory
- General ML that doesn't relate to LLM/Transformer efficiency
- NLP applications (QA, translation, dialogue) that don't focus on efficiency

Return a JSON array of relevant indices, e.g. [0, 2, 5]. Nothing else."""


def _get_llm_client():
    return OpenAI(
        api_key=os.environ.get("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1",
    )


def filter_papers_with_llm(titles):
    """Use MiMo model to filter relevant papers by title."""
    client = _get_llm_client()
    titles_text = "\n".join(f"[{i}] {t}" for i, t in enumerate(titles))

    resp = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[
            {"role": "system", "content": RELEVANCE_PROMPT},
            {"role": "user", "content": titles_text},
        ],
        max_completion_tokens=1024,
        temperature=0.3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = resp.choices[0].message.content.strip()
    start = text.index("[")
    end = text.index("]") + 1
    indices = json.loads(text[start:end])
    return set(indices)


SUMMARY_PROMPT = """你是一个LLM效率优化领域的论文摘要专家。

对于每篇论文，根据其标题和摘要，用一句中文总结：解决了什么问题、提出了什么方法、带来了哪些提升。
要求简洁精炼，不超过60字。

返回JSON对象，key是论文编号，value是中文总结。例如：
{"0": "针对KV cache内存开销问题，提出动态稀疏注意力机制，推理内存降低50%", "2": "..."}

只返回JSON，不要其他内容。"""


def summarize_papers_with_llm(papers):
    """Use MiMo to generate one-line Chinese summaries for papers."""
    client = _get_llm_client()
    papers_text = "\n\n".join(
        f"[{i}] Title: {p.title}\nAbstract: {p.summary[:500]}"
        for i, p in enumerate(papers)
    )

    resp = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": papers_text},
        ],
        max_completion_tokens=4096,
        temperature=0.3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = resp.choices[0].message.content.strip()
    start = text.index("{")
    end = text.rindex("}") + 1
    summaries = json.loads(text[start:end])
    return {int(k): v for k, v in summaries.items()}


def fetch_papers_with_retry(client, search, max_retries=5):
    """Fetch papers with exponential backoff retry on rate limit errors."""
    retry_count = 0
    base_delay = 10  # Start with 10 seconds

    while retry_count < max_retries:
        try:
            papers = []
            for paper in client.results(search):
                papers.append(paper)
                print(f"\rFetching papers... {len(papers)}/{search.max_results}", end="", flush=True)
                time.sleep(0.5)
            print()  # newline after progress
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

    cat_query = "cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.AR OR cat:cs.DC OR cat:cs.PF"
    query = f"({key_query}) AND ({bg_query}) ANDNOT ({exclude_query}) AND ({cat_query})"

    print(query)
    query = query.replace("(", "%28")
    query = query.replace(")", "%29")

    # Construct the default API client.
    client = arxiv.Client()

    # Search for papers in batches
    total_results = 300  # 设置想要获取的总论文数
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

        # First pass: keyword matching
        candidates = []
        for paper in papers:
            date = paper.published.date()
            if date < previous_day:
                continue
            summary = paper.summary
            key_words_has = any(k in summary for k in key_words)
            bg_words_has = any(k in summary for k in bg_words)
            if key_words_has and bg_words_has:
                candidates.append(paper)

        print(f"Keyword matched: {len(candidates)} papers")

        # Second pass: LLM relevance filter
        if candidates:
            titles = [p.title for p in candidates]
            print("Filtering with LLM...")
            try:
                relevant_indices = filter_papers_with_llm(titles)
                filtered = [candidates[i] for i in sorted(relevant_indices) if i < len(candidates)]
                print(f"LLM filter kept: {len(filtered)}/{len(candidates)} papers")
            except Exception as e:
                print(f"LLM filter failed: {e}, falling back to keyword matching only")
                filtered = candidates

            # Generate one-line Chinese summaries
            print("Generating summaries...")
            try:
                summaries = summarize_papers_with_llm(filtered)
                print(f"Generated {len(summaries)} summaries")
            except Exception as e:
                print(f"Summary generation failed: {e}, skipping summaries")
                summaries = {}
        else:
            filtered = []

        # Generate markdown for relevant papers
        for idx, paper in enumerate(filtered):
            title = paper.title
            authors = ", ".join(a.name for a in paper.authors)
            url = paper.entry_id
            date = paper.published.date()
            summary = paper.summary

            # Collect matched keywords as tags (without inline replacement)
            matched_keys = [k for k in key_words if k in summary]
            matched_bgs = [k for k in bg_words if k in summary]
            tags = " ".join(
                [f"![](https://img.shields.io/badge/{k}-F08080)" for k in matched_keys]
                + [f"![](https://img.shields.io/badge/{k}-FF8C00)" for k in matched_bgs]
            )

            print(title)
            markdown_content += f"---\n\n"
            markdown_content += f"## {title}\n\n"
            markdown_content += f"{tags}\n\n" if tags else ""
            one_line = summaries.get(idx, "")
            if one_line:
                markdown_content += f"> {one_line}\n\n"
            markdown_content += f"**Authors:** {authors} | **Date:** {date}\n\n"
            markdown_content += f"**Link:** [{url}]({url})\n\n"
            markdown_content += f"<details><summary>Abstract</summary>\n\n{summary}\n\n</details>\n\n"
            papers_found += 1

    except arxiv.UnexpectedEmptyPageError:
        print(f"\nReached end of results at {papers_found} papers")
    except Exception as e:
        print(f"\nError occurred: {e}")

    print(f"\nTotal papers found: {papers_found}")
    
    if papers_found > 0:
        file_name = f"/Users/xiandong/projects/EfficientPaper/weekly_paper/{today}.md"
        with open(file_name, "w") as wf:
            wf.write(markdown_content)
        

    else:
        print("No new papers found, skipping file creation")

    # Clean up: archive old weekly papers and prune legacy
    cleanup_weekly_papers(today)


def cleanup_weekly_papers(today):
    """Move papers older than 1 month to legacy, delete legacy older than 1 year."""
    dir_root = "/Users/xiandong/projects/EfficientPaper/weekly_paper"
    legacy_dir = "/Users/xiandong/projects/EfficientPaper/docs/weekly_paper/legacy"
    os.makedirs(legacy_dir, exist_ok=True)

    one_month_ago = today - DT.timedelta(days=30)
    one_year_ago = today - DT.timedelta(days=365)

    # Move files older than 1 month from weekly_paper/ to legacy/
    for f in os.listdir(dir_root):
        if not f.endswith('.md'):
            continue
        try:
            file_date = DT.date.fromisoformat(f.replace(".md", ""))
        except ValueError:
            continue
        if file_date < one_month_ago:
            src = os.path.join(dir_root, f)
            dst = os.path.join(legacy_dir, f)
            shutil.move(src, dst)
            print(f"Archived {f} -> legacy/")

    # Delete legacy files older than 1 year
    for f in os.listdir(legacy_dir):
        if not f.endswith('.md'):
            continue
        try:
            file_date = DT.date.fromisoformat(f.replace(".md", ""))
        except ValueError:
            continue
        if file_date < one_year_ago:
            os.remove(os.path.join(legacy_dir, f))
            print(f"Deleted legacy/{f} (older than 1 year)")


if __name__ == "__main__":
    main()
