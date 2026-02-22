# -*- coding: utf-8 -*-
"""
统计 2016–2025 年，每年“疑似构建数据集”的论文数。
逻辑：和 main_loop 保持一致，只做：
  1) arXiv 抓取（fetch_arxiv_recent）
  2) 去重（同一 arxiv_id 只记一次）
  3) 用 BertGate(title, abstract) 过滤
不下载 PDF、不跑 GROBID、不跑 LaTeX。
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from utils.utils_loop import CONFIG, BertGate, fetch_arxiv_recent

START_YEAR = 2016
END_YEAR = 2025

def parse_year(date_str: str) -> Optional[int]:
    """从 arxiv entry 的 published/updated 中解析年份"""
    if not date_str:
        return None
    s = str(date_str).strip()
    if len(s) < 4:
        return None
    try:
        y = int(s[:4])
        return y
    except ValueError:
        return None

def fetch_all_entries_for_cat(cat: str):
    """
    完全照抄你 main_loop 里这一段：
        page = fetch_arxiv_recent(cat, max_results=page_size, start=start)
    只是把结果返回出来。
    """
    entries = []
    start = 0
    page_size = CONFIG.ARXIV_MAX_RESULTS
    while True:
        page = fetch_arxiv_recent(cat, max_results=page_size, start=start)
        if not page:
            break
        entries.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return entries

def main():
    # 为了把 2016–2025 都覆盖上，LAST_N_DAYS 设长一点（~10 年）
    CONFIG.LAST_N_DAYS = 3650

    bert = BertGate()
    seen_ids = set()          # 跨所有类别的全局去重
    year_counter = Counter()  # 只记“数据集论文”数量

    print("CONFIG.CATEGORIES =", CONFIG.CATEGORIES)
    print("LAST_N_DAYS =", CONFIG.LAST_N_DAYS)
    print("BERT_THRESHOLD =", CONFIG.BERT_THRESHOLD)
    print("-" * 60)

    for cat in CONFIG.CATEGORIES:
        entries = fetch_all_entries_for_cat(cat)
        print(f"[{cat}] fetched {len(entries)} entries")

        for e in entries:
            aid = e.get("id")
            if not aid:
                continue

            # 跨类别去重：一个 arxiv_id 只算一次
            if aid in seen_ids:
                continue

            year = parse_year(e.get("published") or e.get("updated"))
            if year is None or not (START_YEAR <= year <= END_YEAR):
                continue

            seen_ids.add(aid)

            # 用和 main_loop 一样的 BERT gate 过滤
            prob = float(bert.score(e["title"], e["abstract"]))
            if prob >= float(CONFIG.BERT_THRESHOLD):
                year_counter[year] += 1

    print("\n=== Dataset-like papers per year (BERT filtered) ===")
    print("year,dataset_paper_count")
    for y in range(START_YEAR, END_YEAR + 1):
        print(y, year_counter[y])

    # 方便后续画图，顺手写一个 csv
    out_dir = Path(CONFIG.BASE_DIR) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dataset_papers_by_year_bert_only.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("year,count\n")
        for y in range(START_YEAR, END_YEAR + 1):
            f.write(f"{y},{year_counter[y]}\n")
    print("\nCSV saved to:", csv_path)

if __name__ == "__main__":
    main()
