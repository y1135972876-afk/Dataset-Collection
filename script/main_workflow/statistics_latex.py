#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：统计给定时间范围内，arXiv 指定类别论文在「标题/摘要」中出现 LaTeX 语法（如 $, \(\), \[\], 常见命令 \alpha 等）的比例。
说明：
- 只做统计，不下载 PDF/源文件；通过解析 arXiv Atom API 返回的条目文本完成。
- 可自由设置类别与时间范围。
- 逐类查询，避免一次查询过大；自动分页直到覆盖时间范围。
- 结果包含：
  * 每个类别的总量、含 LaTeX 数量、比例
  * 合计统计
  * 可选导出 CSV（逐条与聚合两种）

依赖：
    pip install feedparser python-dateutil tqdm
（可选）写 CSV：标准库 csv 即可。

用法示例：
    python arxiv_latex_ratio.py \
        --categories cs.IR cs.DB cs.AI cs.CL cs.CV cs.MA \
        --start 2024-01-01 --end 2024-12-31 \
        --max-results 2000 --sleep 3 \
        --export-csv detailed.csv --export-agg agg.csv

注意：
- arXiv 建议请求频率≤1次/3秒；可通过 --sleep 参数控制。
- 我们用 submittedDate 排序并在本地按发布时间过滤；即便 API 变动，逻辑仍然稳健。
"""

from __future__ import annotations
import argparse
import csv
import time
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Iterable, Tuple

import feedparser
from dateutil import parser as dtparser
from tqdm import tqdm

ARXIV_API = "https://export.arxiv.org/api/query"

# 识别 LaTeX 的简明规则：
# - 行内/行间数学分隔符：$, $$, \( \), \[ \]
# - 常见 TeX 命令：\\alpha, \\beta, \\frac, \\sum, \\int 等
# - 上下标模式：x^{...}, x_{...}
LATEX_PATTERNS = [
    r"\$[^$]+\$",           # $...$
    r"\$\$[^$]+\$\$",      # $$...$$
    r"\\\([^)]*\\\)",     # \( ... \)
    r"\\\[[^\]]*\\\]",   # \[ ... \]
    r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b",
    r"\\(frac|sum|int|prod|lim|log|sin|cos|tan|mathbf|mathbb|mathcal|mathrm|left|right)\b",
    r"\^[{\[]?.+?[}\]]?",  # 上标粗略捕获
    r"_[{\[]?.+?[}\]]?",   # 下标粗略捕获
]
LATEX_REGEX = re.compile("(" + ")|(".join(LATEX_PATTERNS) + ")", re.IGNORECASE | re.DOTALL)

@dataclass
class Paper:
    id: str
    title: str
    summary: str
    published: datetime
    category: str


def parse_args() -> argparse.Namespace:
    # 计算默认时间范围：过去 365 天（含）到今天（含），UTC
    today = datetime.utcnow().date()
    start_default = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_default = today.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="统计 arXiv 指定类别在时间段内含 LaTeX 的比例")
    parser.add_argument("--categories", nargs="+", default=["cs.IR", "cs.DB", "cs.AI", "cs.CL", "cs.CV", "cs.MA"],
                        help="arXiv 类别列表，例如 cs.AI cs.CL（默认：cs.IR cs.DB cs.AI cs.CL cs.CV cs.MA）")
    parser.add_argument("--start", type=str, default=start_default, help=f"起始日期 YYYY-MM-DD（含），默认：{start_default}")
    parser.add_argument("--end", type=str, default=end_default, help=f"结束日期 YYYY-MM-DD（含），默认：{end_default}")
    parser.add_argument("--max-results", type=int, default=2000, help="每次分页条数（建议≤2000，默认 2000）")
    parser.add_argument("--sleep", type=float, default=3.0, help="分页请求之间的休眠秒数，遵守 arXiv 频率要求（默认 3 秒）")
    parser.add_argument("--export-csv", type=str, default=None, help="导出逐条结果到 CSV 文件路径（默认不导出）")
    parser.add_argument("--export-agg", type=str, default=None, help="导出聚合统计到 CSV 文件路径（默认不导出）")
    return parser.parse_args()


def dt_utc(date_str: str) -> datetime:
    dt = dtparser.parse(date_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def within_range(dt: datetime, start: datetime, end: datetime) -> bool:
    return start <= dt <= end


def has_latex(text: str) -> bool:
    if not text:
        return False
    return LATEX_REGEX.search(text) is not None


def build_query(category: str) -> str:
    # 只按类别过滤；排序按 submittedDate 升序，便于边拉边过滤
    return f"search_query=cat:{category}&sortBy=submittedDate&sortOrder=ascending"


def fetch_category_papers(category: str, start_dt: datetime, end_dt: datetime, page_size: int, sleep_sec: float) -> List[Paper]:
    papers: List[Paper] = []
    start_index = 0

    pbar = tqdm(desc=f"Fetching {category}")
    while True:
        url = f"{ARXIV_API}?{build_query(category)}&start={start_index}&max_results={page_size}"
        feed = feedparser.parse(url)
        # 安全性：若 API 临时失败，返回可能为空；做健壮处理
        entries = getattr(feed, "entries", [])
        if not entries:
            break

        new_count = 0
        for e in entries:
            # published/updated 以 published 为准（submitted 时间）；若缺失则跳过
            if not hasattr(e, "published"):
                continue
            pdt = dt_utc(e.published)
            if pdt > end_dt:
                # 还未到达时间上限，继续（因为升序，我们会在未来条目更大，仍需继续分页）
                pass
            if pdt < start_dt:
                # 小于最小日期，继续看后面的（升序，后面会更晚）
                continue
            # 过滤在范围内
            cat = category
            papers.append(Paper(
                id=e.get("id", ""),
                title=e.get("title", ""),
                summary=e.get("summary", ""),
                published=pdt,
                category=cat,
            ))
            new_count += 1
        pbar.update(new_count)

        # 终止条件：若最后一条的发布时间已经超过 end_dt 且新增为 0，说明已越界
        last_dt = dt_utc(entries[-1].published) if hasattr(entries[-1], "published") else None
        if last_dt and last_dt > end_dt and new_count == 0:
            break

        start_index += page_size
        time.sleep(sleep_sec)
    pbar.close()
    return papers


def aggregate(papers: List[Paper]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    for p in papers:
        k = p.category
        if k not in agg:
            agg[k] = {"total": 0, "latex": 0}
        agg[k]["total"] += 1
        text = f"{p.title}\n{p.summary}"
        if has_latex(text):
            agg[k]["latex"] += 1
    return agg


def print_report(agg: Dict[str, Dict[str, float]]) -> None:
    print("\n===== 分类统计 =====")
    grand_total = 0
    grand_latex = 0
    for cat in sorted(agg.keys()):
        total = int(agg[cat]["total"]) or 0
        latex = int(agg[cat]["latex"]) or 0
        ratio = (latex / total * 100) if total else 0.0
        grand_total += total
        grand_latex += latex
        print(f"{cat:6s}  总数: {total:6d}  含LaTeX: {latex:6d}  比例: {ratio:6.2f}%")
    print("\n===== 合计 =====")
    overall = (grand_latex / grand_total * 100) if grand_total else 0.0
    print(f"总数: {grand_total}  含LaTeX: {grand_latex}  比例: {overall:.2f}%")


def export_detailed_csv(papers: List[Paper], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "published", "has_latex", "title", "summary"])
        for p in papers:
            text = f"{p.title}\n{p.summary}"
            writer.writerow([
                p.id,
                p.category,
                p.published.isoformat(),
                int(has_latex(text)),
                p.title.replace("\n", " ").strip(),
                p.summary.replace("\n", " ").strip(),
            ])
    print(f"已导出逐条 CSV: {path}")


def export_agg_csv(agg: Dict[str, Dict[str, float]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total", "latex", "ratio_percent"])
        for cat in sorted(agg.keys()):
            total = int(agg[cat]["total"]) or 0
            latex = int(agg[cat]["latex"]) or 0
            ratio = (latex / total * 100) if total else 0.0
            writer.writerow([cat, total, latex, f"{ratio:.2f}"])
    print(f"已导出聚合 CSV: {path}")


def main():
    args = parse_args()

    start_dt = dt_utc(args.start + "T00:00:00Z")
    end_dt = dt_utc(args.end + "T23:59:59Z")

    all_papers: List[Paper] = []
    for cat in args.categories:
        ps = fetch_category_papers(cat, start_dt, end_dt, args.max_results, args.sleep)
        all_papers.extend(ps)

    agg = aggregate(all_papers)
    print_report(agg)

    if args.export_csv:
        export_detailed_csv(all_papers, args.export_csv)
    if args.export_agg:
        export_agg_csv(agg, args.export_agg)


if __name__ == "__main__":
    main()
