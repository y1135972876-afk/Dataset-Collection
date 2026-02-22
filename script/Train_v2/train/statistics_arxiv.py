# -*- coding: utf-8 -*-
"""
只统计 2025 年（至今）的 arXiv 论文：
  - 按类别统计当年总篇数（通过翻页累加，不再信任“Total of N entries”）
  - 对 2025 年内所有论文的 (title, abstract) 逐篇跑 BertGate
  - 统计通过 BERT gate 的篇数
  - 同时把通过 BERT 的论文保存成 JSON

依赖：
    pip install requests beautifulsoup4 urllib3 certifi

工程依赖：
    from utils.utils_loop import CONFIG, BertGate
        - BertGate.score(title, abstract) -> float 概率
        - CONFIG.BERT_THRESHOLD（可选，若没有则用默认 0.5）

使用前请替换 USER_AGENT_EMAIL 为你自己的邮箱。
"""

import os
import re
import json
import time
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.utils_loop import CONFIG, BertGate  # 路径按你工程实际调整


# ========================= 基本配置 =========================

BASE_URL = "https://arxiv.org"

# 要爬取的类别
CATEGORIES = [
    "cs.IR",
    "cs.DB",
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.MA",
]

# 只统计 2025 年（至今）
YEAR = 2025

# 每页拉多少篇（只是单页上限，不会漏数据；
# 如果怕页面太大，可以改小一点，比如 1000 或 500）
PAGE_SIZE = 2000

# arXiv UA 要求必须带邮箱
USER_AGENT_EMAIL = "your_email@example.com"  # TODO: 换成你自己的邮箱

# 输出目录：保存通过 BERT gate 的论文 JSON
OUTPUT_DIR = "./arxiv_bert_filtered_2025"


# ========================= HTTP Session =========================

def make_session() -> requests.Session:
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update({
        "User-Agent": f"kzlab-arxiv-stats/0.1 (mailto:{USER_AGENT_EMAIL})"
    })
    return s


SESSION = make_session()

# ========================= BERT Gate =========================

bert = BertGate()
BERT_THRESHOLD = float(getattr(CONFIG, "BERT_THRESHOLD", 0.5))


# ========================= 工具函数 =========================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def year_to_list_code(year: int) -> str:
    """
    arXiv 列表页格式：
        https://arxiv.org/list/cs.AI/25   -> 2025 年（至今）
    """
    return str(year)[-2:]


# ========================= 爬每一页 + BERT 过滤 =========================

def fetch_papers_from_list_page(
    list_url: str,
    category: str,
    skip_argument: int
) -> Tuple[List[Dict], int, int]:
    """
    给定一个分类某一页的 list URL，解析该页的所有论文：
    - 访问这一页里所有论文的 abstract 页（dt/dd 全部遍历）
    - 对每篇 (title, abstract) 走一次 BertGate
    - 返回：
        papers: 通过 BERT gate 的论文列表（这一页）
        total_entries_this_page: 这一页实际遍历的条目数
        bert_kept_this_page: 这一页通过 BERT 的条目数
    """
    try:
        resp = SESSION.get(list_url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERR] 请求 list 页失败: {e} ({list_url})")
        return [], 0, 0

    soup = BeautifulSoup(resp.content, "html.parser")
    dls = soup.find_all("dl")
    if not dls:
        print(f"[WARN] list 页中未找到 <dl>：{list_url}")
        return [], 0, 0

    recent_submissions = dls[0]

    dt_list = recent_submissions.find_all("dt")
    dd_list = recent_submissions.find_all("dd")

    total_entries_this_page = len(dt_list)
    bert_kept_this_page = 0
    papers: List[Dict] = []

    print(f"[INFO] {category} list page {list_url} 条目数: {total_entries_this_page}")

    for idx, (dt, dd) in enumerate(zip(dt_list, dd_list)):
        try:
            abstract_link = dt.find("a", title="Abstract")
            if not abstract_link or not abstract_link.get("href"):
                print(f"[WARN] 找不到 Abstract 链接，跳过一条 ({category}, idx={idx})")
                continue

            paper_url = BASE_URL + abstract_link["href"]

            # 打开 abstract 页（用带 retry 的 SESSION）
            try:
                paper_resp = SESSION.get(paper_url, timeout=60)
                paper_resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERR] 请求 abstract 页失败，跳过该论文: {e} ({paper_url})")
                continue

            paper_soup = BeautifulSoup(paper_resp.content, "html.parser")

            # 1. HTML / PDF URL
            html_link = dt.find("a", title="View HTML")
            if html_link and html_link.get("href"):
                html_url = html_link["href"]
                if html_url.startswith("/"):
                    html_url = BASE_URL + html_url
            else:
                pdf_url = dt.find("a", title="Download PDF")
                if pdf_url and pdf_url.get("href"):
                    html_url = BASE_URL + pdf_url["href"]
                else:
                    html_url = paper_url  # 兜底：用 abstract 页 URL

            # 2. 标题
            title_elem = paper_soup.find("h1", class_="title")
            if not title_elem:
                title_elem = paper_soup.find("h1", class_="title mathjax")
            title = title_elem.text.strip().replace("Title:", "").strip() if title_elem else ""

            # 3. 摘要
            abs_elem = paper_soup.find("blockquote", class_="abstract")
            if not abs_elem:
                abs_elem = paper_soup.find("blockquote", class_="abstract mathjax")
            abstract = abs_elem.text.strip().replace("Abstract:", "").strip() if abs_elem else ""

            # ========= 用 BERT gate 过滤 title + abstract（这一页所有条目都跑） =========
            try:
                prob = bert.score(title, abstract)
            except Exception as be:
                print(f"[BERT-ERR] [{category}] idx={idx}, 标题片段='{title[:50]}...': {be}")
                prob = 0.0

            if prob < BERT_THRESHOLD:
                # 不像“发数据集”的论文，跳过
                continue

            # 通过 BERT gate
            bert_kept_this_page += 1

            # 4. 作者
            authors_elem = paper_soup.find("div", class_="authors")
            authors = authors_elem.text.strip().replace("Authors:", "").strip() if authors_elem else ""

            # 5. 日期
            dateline_elem = paper_soup.find("div", class_="dateline")
            if dateline_elem:
                dateline = dateline_elem.text.strip().replace("Date:", "").strip()
                date_match = re.search(r"Submitted on (\d+ \w+ \d+)", dateline)
                date_str = date_match.group(1) if date_match else dateline
            else:
                date_str = "N/A"

            # 6. DOI
            doi_elem = paper_soup.find("td", class_="tablecell arxivdoi")
            if doi_elem and doi_elem.find("a"):
                doi = doi_elem.find("a")["href"]
            else:
                doi = "N/A"

            global_index = skip_argument + idx  # 全局序号（从这一年的开头算起）

            papers.append({
                "global_index": global_index,
                "category": category,
                "url": html_url,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "date": date_str,
                "doi": doi,
                "bert_prob": float(prob),
                "has_dataset_like_signal": True,
                "Text that mentions Dataset": None,
                "Dataset Links": None,
            })

            print(f"[BERT-KEPT] [{category}] #{global_index} prob={prob:.3f} title={title[:80]}")

            # 为了对 arXiv 友好，稍微睡一下（可视情况调）
            time.sleep(0.2)

        except Exception as e:
            print(f"[ERR] 解析单篇论文出错 ({category}, idx={idx}): {e}")

    print(
        f"[STATS-PAGE] {category} page skip={skip_argument}: "
        f"total={total_entries_this_page}, bert_kept={bert_kept_this_page}"
    )
    return papers, total_entries_this_page, bert_kept_this_page


# ========================= 主逻辑：只按 2025 年 + 类统计 =========================

def main():
    ensure_dir(OUTPUT_DIR)

    year_code = year_to_list_code(YEAR)
    print(f"\n========== 处理年份 {YEAR} (list code: {year_code}) ==========")

    grand_total_entries = 0
    grand_bert_kept = 0

    for category in CATEGORIES:
        print(f"\n---- 类别 {category} ----")

        skip_argument = 0
        total_entries_for_cat = 0
        bert_kept_total = 0
        bert_papers_all: List[Dict] = []

        while True:
            list_url = (
                f"{BASE_URL}/list/{category}/{year_code}"
                f"?skip={skip_argument}&show={PAGE_SIZE}"
            )
            papers, total_page, bert_page = fetch_papers_from_list_page(
                list_url=list_url,
                category=category,
                skip_argument=skip_argument,
            )

            # 当前页没有内容了，说明翻完了
            if total_page == 0:
                break

            total_entries_for_cat += total_page
            bert_kept_total += bert_page
            bert_papers_all.extend(papers)

            # 下一页：这里用 total_page 来推进，防止 "总是 50 < PAGE_SIZE" 被误判
            skip_argument += total_page

            # 如果这一页条目数 < PAGE_SIZE，说明已经是最后一页
            if total_page < PAGE_SIZE:
                break

        grand_total_entries += total_entries_for_cat
        grand_bert_kept += bert_kept_total

        print(
            f"[SUMMARY] {category} {YEAR}: "
            f"total={total_entries_for_cat}, "
            f"bert_kept={bert_kept_total}, "
            f"ratio={bert_kept_total / max(total_entries_for_cat, 1):.4f}"
        )

        # 写 JSON（只写通过 BERT gate 的论文）
        out_dir_year = os.path.join(OUTPUT_DIR, str(YEAR))
        ensure_dir(out_dir_year)
        out_path = os.path.join(out_dir_year, f"{category.replace('.', '_')}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bert_papers_all, f, ensure_ascii=False, indent=2)

        print(
            f"[SAVE] {category} {YEAR} 通过 BERT 的论文共 "
            f"{bert_kept_total} 篇，已写入 {out_path}"
        )

    # 整体汇总（所有类别合在一起）
    print(
        f"\n========== 汇总 {YEAR} ==========\n"
        f"所有类别总篇数: {grand_total_entries}\n"
        f"通过 BERT gate 的总篇数: {grand_bert_kept}\n"
        f"总体比例: {grand_bert_kept / max(grand_total_entries, 1):.4f}\n"
    )


if __name__ == "__main__":
    main()
