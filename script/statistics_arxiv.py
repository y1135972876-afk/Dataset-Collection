# -*- coding: utf-8 -*-
"""
按年份 + 分类统计 arXiv 论文总数 & 通过 BertGate 的论文数，
并把通过 BERT gate 的论文保存到 JSON 文件。

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
from typing import List, Dict, Optional, Tuple

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

# 要统计的年份（可以写多个）
YEARS = [2025]

# 每页拉多少篇（arXiv 最多 2000，列表页支持 show=2000）
PAGE_SIZE = 2000

# arXiv UA 要求必须带邮箱
USER_AGENT_EMAIL = "your_email@example.com"  # TODO: 换成你自己的邮箱

# 输出目录：保存通过 BERT gate 的论文 JSON
OUTPUT_DIR = "./arxiv_bert_filtered"


# ========================= HTTP Session =========================

def make_session() -> requests.Session:
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "GET"]
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
        https://arxiv.org/list/cs.AI/23      -> 2023 全年
        https://arxiv.org/list/cs.AI/2401    -> 2024 年 1 月
    这里我们用“全年”的两位年份码。
    """
    return str(year)[-2:]


# ========================= 统计总篇数 =========================

def get_entries_count(list_url: str) -> Optional[int]:
    """
    从 arXiv list 页顶部的 “Total of N entries” 里解析出总篇数 N。

    返回:
        int: 总篇数
        None: 解析失败或请求失败
    """
    try:
        resp = SESSION.get(list_url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERR] get_entries_count 请求失败: {e} ({list_url})")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 在 HTML 中找类似 “Total of 1234 entries”
    possible_elements = soup.find_all(string=re.compile(r"Total of \d+ entries"))
    if not possible_elements:
        print(f"[WARN] 未找到 'Total of N entries' 文本: {list_url}")
        return None

    entries_info = possible_elements[0].strip()
    m = re.search(r"of\s+(\d+)\s+entries", entries_info)
    if not m:
        print(f"[WARN] 无法从 '{entries_info}' 解析出 entries 数量")
        return None

    count = int(m.group(1))
    return count


# ========================= 爬每一篇详情 + BERT 过滤 =========================

def fetch_papers_from_list_page(
    list_url: str,
    category: str,
    skip_argument: int
) -> Tuple[List[Dict], int, int]:
    """
    给定一个分类某一页的 list URL，解析该页的所有论文：
    - 访问每篇的 abstract 页，提取 title / abstract 等
    - 用 BertGate 对 (title, abstract) 打分过滤
    - 返回：
        papers: 通过 BERT gate 的论文列表
        total_entries_this_page: 这一页实际遍历的条目数
        bert_kept_this_page: 这一页通过 BERT 的条目数
    """
    try:
        resp = SESSION.get(list_url, timeout=20)
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

            # 打开 abstract 页
            paper_resp = SESSION.get(paper_url, timeout=10)
            paper_resp.raise_for_status()
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

            # ========= 用第一个 BERT gate 过滤 title + abstract =========
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

            # 为了对 arXiv 友好，稍微睡一下
            time.sleep(0.2)

        except Exception as e:
            print(f"[ERR] 解析单篇论文出错 ({category}, idx={idx}): {e}")

    print(f"[STATS-PAGE] {category} page skip={skip_argument}: total={total_entries_this_page}, bert_kept={bert_kept_this_page}")
    return papers, total_entries_this_page, bert_kept_this_page


# ========================= 主逻辑：按年 + 类统计 =========================

def main():
    ensure_dir(OUTPUT_DIR)

    for year in YEARS:
        year_code = year_to_list_code(year)
        print(f"\n========== 处理年份 {year} (list code: {year_code}) ==========")

        for category in CATEGORIES:
            print(f"\n---- 类别 {category} ----")

            # 1. 先拿这一年这个类别的总篇数
            list_url_for_count = f"{BASE_URL}/list/{category}/{year_code}"
            total_entries = get_entries_count(list_url_for_count)
            if total_entries is None:
                print(f"[SKIP] 无法获取 {category} {year} 的总篇数，跳过。")
                continue

            print(f"[TOTAL] {category} {year}: {total_entries} 篇（列表页统计）")

            # 2. 遍历这一年该类别所有 list 页，做 BERT 过滤
            skip_argument = 0
            bert_kept_total = 0
            bert_papers_all: List[Dict] = []

            while skip_argument < total_entries:
                list_url = f"{BASE_URL}/list/{category}/{year_code}?skip={skip_argument}&show={PAGE_SIZE}"
                papers, total_page, bert_page = fetch_papers_from_list_page(
                    list_url=list_url,
                    category=category,
                    skip_argument=skip_argument,
                )

                bert_papers_all.extend(papers)
                bert_kept_total += bert_page

                skip_argument += PAGE_SIZE
                # 若这一页实际条目数少于 PAGE_SIZE，说明已经到末尾，也可以 break
                if total_page < PAGE_SIZE:
                    break

            print(f"[SUMMARY] {category} {year}: total={total_entries}, bert_kept={bert_kept_total}, ratio={bert_kept_total/max(total_entries,1):.4f}")

            # 3. 写 JSON（只写通过 BERT gate 的论文）
            out_dir_year = os.path.join(OUTPUT_DIR, str(year))
            ensure_dir(out_dir_year)
            out_path = os.path.join(out_dir_year, f"{category.replace('.', '_')}.json")

            # 写成一整个 list，方便直接用 json.load 读回
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bert_papers_all, f, ensure_ascii=False, indent=2)

            print(f"[SAVE] {category} {year} 通过 BERT 的论文共 {bert_kept_total} 篇，已写入 {out_path}")


if __name__ == "__main__":
    main()
