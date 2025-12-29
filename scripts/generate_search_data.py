#!/usr/bin/env python3
"""
生成论文搜索数据 JSON 文件
供独立搜索页面使用
"""

import json
import os
import sys

import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_paper_pb2 as eppb
from scripts.generate_paper_list import word_pb2str

# 关键词映射
KEYWORD_MAP = word_pb2str


def read_all_papers():
    """读取所有论文元数据"""
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
    """生成搜索数据 JSON"""
    papers = read_all_papers()

    data = []
    venues = set()
    keywords = set()
    years = set()

    for pinfo, filename, year in papers:
        paper = pinfo.paper
        pub = pinfo.pub

        # 提取关键词
        paper_keywords = []
        for word in pinfo.keyword.words:
            if word in KEYWORD_MAP:
                kw = KEYWORD_MAP[word]
                paper_keywords.append(kw)
                keywords.add(kw)

        venue = pub.where if pub.where else "arXiv"
        venues.add(venue)
        years.add(year)

        # 处理封面图片路径
        cover_url = None
        if pinfo.cover.url:
            file_name = filename.replace(".prototxt", "")
            # 检查可能的图片路径
            possible_paths = [
                f"./notes/{pinfo.cover.url}",
                f"./notes/{year}/{pinfo.cover.url}",
                f"./notes/{year}/{file_name}/{pinfo.cover.url}",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    # 转换为相对于 docs 目录的路径
                    cover_url = path.replace("./", "../")
                    break

        # 构建论文数据
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
        }
        data.append(paper_data)

    # 按年份降序、标题升序排序
    data.sort(key=lambda x: (-x["year"], x["title"].lower()))

    # 生成完整的搜索数据结构
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

    # 写入 JSON 文件
    output_path = "./docs/js/papers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(search_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {output_path} with {search_data['total']} papers")


if __name__ == "__main__":
    main()
