#!/usr/bin/env python3
"""
Generate paper search data JSON file
For use by standalone search page
"""

import json
import os
import sys

import google.protobuf as pb
import google.protobuf.text_format

# Add project root to path before importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import efficient_paper_pb2 as eppb
from scripts.generate_paper_list import word_pb2str

# Keyword mapping
KEYWORD_MAP = word_pb2str


def read_all_papers():
    """Read all paper metadata"""
    papers = []
    meta_dir = "./meta"

    for year_dir in os.listdir(meta_dir):
        if year_dir == "search":
            continue
        year_path = os.path.join(meta_dir, year_dir)
        if not os.path.isdir(year_path):
            continue

        try:
            year = int(year_dir)
        except ValueError:
            continue

        for filename in os.listdir(year_path):
            if not filename.endswith(".prototxt"):
                continue
            filepath = os.path.join(year_path, filename)
            pinfo = eppb.PaperInfo()
            try:
                with open(filepath, "r") as f:
                    pb.text_format.Merge(f.read(), pinfo)
                papers.append((pinfo, filename, year))
            except Exception as e:
                print(f"Warning: Failed to read {filepath}: {e}", file=sys.stderr)

    return papers


def generate_search_data():
    """Generate search data JSON"""
    papers = read_all_papers()

    data = []
    venues = set()
    keywords = set()
    years = set()

    for pinfo, filename, year in papers:
        paper = pinfo.paper
        pub = pinfo.pub

        # Extract keywords
        paper_keywords = []
        for word in pinfo.keyword.words:
            if word in KEYWORD_MAP:
                kw = KEYWORD_MAP[word]
                paper_keywords.append(kw)
                keywords.add(kw)

        venue = pub.where if pub.where else "arXiv"
        venues.add(venue)
        years.add(year)

        # Process cover image path
        cover_url = None
        if pinfo.cover.url:
            file_name = filename.replace(".prototxt", "")
            # Check possible image paths
            possible_paths = [
                f"./notes/{pinfo.cover.url}",
                f"./notes/{year}/{pinfo.cover.url}",
                f"./notes/{year}/{file_name}/{pinfo.cover.url}",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    # Use absolute path from site root (notes/ is at root level)
                    cover_url = path.replace("./", "")
                    break

        # Process note URL path
        note_url = None
        file_name = filename.replace(".prototxt", "")
        # Check if note file exists (note.md)
        note_md_path = f"./notes/{year}/{file_name}/note.md"
        if os.path.exists(note_md_path):
            # Use absolute path from site root (notes/ is at root level)
            note_url = f"notes/{year}/{file_name}/note/"

        # Build paper data
        paper_data = {
            "id": filename.replace(".prototxt", ""),
            "title": paper.title,
            "abbr": paper.abbr,
            "url": paper.url,
            "authors": list(paper.authors),
            "institutions": list(paper.institutions),
            "year": year,
            "venue": venue,
            "cover": cover_url,
            "keywords": paper_keywords,
            "code_url": pinfo.code.url if pinfo.code.url else None,
            "note_url": note_url,
            "prototxt_path": f"meta/{year}/{filename}",
            "update_time": pinfo.update_time if pinfo.HasField('update_time') else 0,
        }
        data.append(paper_data)

    # Sort by update_time descending (most recently updated first)
    data.sort(key=lambda x: (-x["update_time"], x["title"].lower()))

    # Generate complete search data structure
    search_data = {
        "papers": data,
        "filters": {
            "years": sorted(years, reverse=True),
            "venues": sorted(venues),
            "keywords": sorted(keywords),
        },
        "total": len(data),
    }

    return search_data


def main():
    search_data = generate_search_data()

    # Write JSON file
    output_path = "./docs/js/papers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(search_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {output_path} with {search_data['total']} papers")


if __name__ == "__main__":
    main()
