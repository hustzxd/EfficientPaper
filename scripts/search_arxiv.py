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


REPO_ROOT = "/Users/xiandong/projects/EfficientPaper"
WEEKLY_DIR = os.path.join(REPO_ROOT, "weekly_paper")
LEGACY_DIR = os.path.join(REPO_ROOT, "docs", "weekly_paper", "legacy")
SEARCH_WORDS_PATH = os.path.join(REPO_ROOT, "meta", "search", "efficient_keywords.prototxt")


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default, minimum=1):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using {default}")
        return default
    if parsed < minimum:
        print(f"Invalid {name}={value!r}; using {default}")
        return default
    return parsed


def env_float(name, default, minimum=0.0):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using {default}")
        return default
    if parsed < minimum:
        print(f"Invalid {name}={value!r}; using {default}")
        return default
    return parsed


def arxiv_abs_query(words):
    terms = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if " " in word:
            word = '"' + word.replace('"', '\\"') + '"'
        terms.append(f"abs:{word}")
    return " OR ".join(terms)


def build_arxiv_query(key_words, bg_words, exclude_words, previous_day, today):
    key_query = arxiv_abs_query(key_words)
    bg_query = arxiv_abs_query(bg_words)
    exclude_query = arxiv_abs_query(exclude_words)

    if not key_query or not bg_query:
        raise ValueError("Search keywords and background words must both be configured.")

    cat_query = "cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.AR OR cat:cs.DC OR cat:cs.PF"
    submitted_query = (
        f"submittedDate:[{previous_day.strftime('%Y%m%d')}0000 TO "
        f"{today.strftime('%Y%m%d')}2359]"
    )
    query_parts = [
        f"({key_query})",
        f"({bg_query})",
        f"({cat_query})",
        submitted_query,
    ]
    query = " AND ".join(query_parts)
    if exclude_query:
        query = f"{query} ANDNOT ({exclude_query})"
    return query


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
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise RuntimeError("MIMO_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1",
    )


def filter_papers_with_llm(titles):
    """Use MiMo model to filter relevant papers by title."""
    client = _get_llm_client()
    titles_text = "\n".join(f"[{i}] {t}" for i, t in enumerate(titles))

    resp = client.chat.completions.create(
        model="mimo-v2.5",
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
        model="mimo-v2.5",
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


def is_limited_http_error(error):
    text = str(error).lower()
    return (
        "429" in text
        or "403" in text
        or "rate limit" in text
        or "too many" in text
        or "forbidden" in text
        or "restricted" in text
    )


def fetch_papers_with_retry(client, search, max_retries=3, base_delay=60):
    """Fetch papers with exponential backoff retry on arXiv rate/IP limits."""
    retry_count = 0

    while retry_count <= max_retries:
        try:
            papers = []
            for paper in client.results(search):
                papers.append(paper)
                print(f"\rFetching papers... {len(papers)}/{search.max_results}", end="", flush=True)
            print()  # newline after progress
            return papers
        except arxiv.UnexpectedEmptyPageError:
            print("\nReached end of arXiv results")
            return papers
        except arxiv.HTTPError as e:
            if is_limited_http_error(e):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"\nMax retries ({max_retries}) reached for arXiv rate/IP limit. Giving up.")
                    raise

                delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                print(
                    f"\narXiv rate/IP limit hit. Retrying in {delay} seconds... "
                    f"(attempt {retry_count}/{max_retries})"
                )
                time.sleep(delay)
            else:
                raise

    return []


def main():
    today = DT.date.today()
    files = [f for f in os.listdir(WEEKLY_DIR) if f.endswith('.md')]
    files.sort()
    if not files:
        print(f"No previous weekly paper found in {WEEKLY_DIR}", file=sys.stderr)
        return 1

    previous_day = files[-1].replace(".md", "")
    previous_day = DT.date.fromisoformat(previous_day)
    print(f"Previous update date: {previous_day}")
    if today == previous_day:
        print("Already up to date")
        return 0

    search_words = eppb.SearchWord()
    try:
        with open(SEARCH_WORDS_PATH, "r", encoding="utf-8") as rf:
            pb.text_format.Merge(rf.read(), search_words)
    except Exception as e:
        print(f"Failed to read search words from {SEARCH_WORDS_PATH}: {e}", file=sys.stderr)
        return 1

    bg_words = list(search_words.background_words)
    key_words = list(search_words.key_words)
    exclude_words = list(search_words.exclude_words)

    try:
        query = build_arxiv_query(key_words, bg_words, exclude_words, previous_day, today)
    except ValueError as e:
        print(f"Invalid weekly paper search configuration: {e}", file=sys.stderr)
        return 1

    print(query)

    total_results = env_int("WEEKLY_PAPER_MAX_RESULTS", 100)
    arxiv_delay = env_float("WEEKLY_PAPER_ARXIV_DELAY_SECONDS", 15.0)
    arxiv_page_size = env_int("WEEKLY_PAPER_ARXIV_PAGE_SIZE", 100)
    arxiv_retries = env_int("WEEKLY_PAPER_ARXIV_RETRIES", 2, minimum=0)
    arxiv_retry_base_delay = env_int("WEEKLY_PAPER_ARXIV_RETRY_BASE_DELAY", 300, minimum=0)
    use_llm = env_bool("WEEKLY_PAPER_USE_LLM", True)

    print(
        "arXiv fetch config: "
        f"max_results={total_results}, page_size={arxiv_page_size}, "
        f"delay_seconds={arxiv_delay}, retries={arxiv_retries}"
    )
    print(f"LLM filtering/summaries: {'enabled' if use_llm else 'disabled'}")

    client = arxiv.Client(
        page_size=arxiv_page_size,
        delay_seconds=arxiv_delay,
        num_retries=0,
    )
    markdown_content = f"# {today}\n\n"
    papers_found = 0

    search = arxiv.Search(
        query=query,
        max_results=total_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    try:
        print("Fetching papers from arXiv...")
        papers = fetch_papers_with_retry(
            client,
            search,
            max_retries=arxiv_retries,
            base_delay=arxiv_retry_base_delay,
        )
        print(f"Successfully fetched {len(papers)} papers")
    except Exception as e:
        print(f"\nFailed to fetch papers from arXiv: {e}", file=sys.stderr)
        print("Skip weekly paper generation and cleanup for this run.", file=sys.stderr)
        return 1

    candidates = []
    for paper in papers:
        date = paper.published.date()
        if date < previous_day:
            continue
        summary = paper.summary
        summary_lower = summary.lower()
        key_words_has = any(k.lower() in summary_lower for k in key_words)
        bg_words_has = any(k.lower() in summary_lower for k in bg_words)
        if key_words_has and bg_words_has:
            candidates.append(paper)

    print(f"Keyword matched: {len(candidates)} papers")

    summaries = {}
    if candidates and use_llm:
        titles = [p.title for p in candidates]
        print("Filtering with LLM...")
        try:
            relevant_indices = filter_papers_with_llm(titles)
            filtered = [candidates[i] for i in sorted(relevant_indices) if i < len(candidates)]
            print(f"LLM filter kept: {len(filtered)}/{len(candidates)} papers")
        except Exception as e:
            print(f"LLM filter failed: {e}, falling back to keyword matching only")
            filtered = candidates

        if filtered:
            print("Generating summaries...")
            try:
                summaries = summarize_papers_with_llm(filtered)
                print(f"Generated {len(summaries)} summaries")
            except Exception as e:
                print(f"Summary generation failed: {e}, skipping summaries")
    elif candidates:
        print("Skipping LLM filter and summaries because WEEKLY_PAPER_USE_LLM is disabled.")
        filtered = candidates
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
        summary_lower = summary.lower()
        matched_keys = [k for k in key_words if k.lower() in summary_lower]
        matched_bgs = [k for k in bg_words if k.lower() in summary_lower]
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

    print(f"\nTotal papers found: {papers_found}")

    if papers_found > 0:
        file_name = os.path.join(WEEKLY_DIR, f"{today}.md")
        with open(file_name, "w", encoding="utf-8") as wf:
            wf.write(markdown_content)
    else:
        print("No new papers found, skipping file creation")

    # Clean up: archive old weekly papers and prune legacy
    cleanup_weekly_papers(today)
    return 0


def cleanup_weekly_papers(today):
    """Move papers older than 1 month to legacy, delete legacy older than 1 year."""
    os.makedirs(LEGACY_DIR, exist_ok=True)

    one_month_ago = today - DT.timedelta(days=30)
    one_year_ago = today - DT.timedelta(days=365)

    # Move files older than 1 month from weekly_paper/ to legacy/
    for f in os.listdir(WEEKLY_DIR):
        if not f.endswith('.md'):
            continue
        try:
            file_date = DT.date.fromisoformat(f.replace(".md", ""))
        except ValueError:
            continue
        if file_date < one_month_ago:
            src = os.path.join(WEEKLY_DIR, f)
            dst = os.path.join(LEGACY_DIR, f)
            shutil.move(src, dst)
            print(f"Archived {f} -> legacy/")

    # Delete legacy files older than 1 year
    for f in os.listdir(LEGACY_DIR):
        if not f.endswith('.md'):
            continue
        try:
            file_date = DT.date.fromisoformat(f.replace(".md", ""))
        except ValueError:
            continue
        if file_date < one_year_ago:
            os.remove(os.path.join(LEGACY_DIR, f))
            print(f"Deleted legacy/{f} (older than 1 year)")


if __name__ == "__main__":
    raise SystemExit(main())
