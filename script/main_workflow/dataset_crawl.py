import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------- 基础配置 --------------------
base_url = "https://arxiv.org"

categories = [
    "cs.IR",
    "cs.DB",
    "cs.AI", #
    "cs.CL", #
    "cs.CV", #
    "cs.MA",
]

# 这里设置日期入口：不传参数默认今天；想指定就填年月日
def set_date(year=None, month=None, day=None):
    if all(x is None for x in (year, month, day)):
        dt = datetime.now()
    else:
        dt = datetime(year, month, day)
    return dt.strftime("%Y-%m-%d"), dt

# today_str 用来拼路径；today_dt 用来格式化网页里的日期标题
# today, today_dt = set_date()               # 默认今天
today, today_dt = set_date(2025, 9, 17)   # 想指定哪天就用这一行

# -------------------- 工具函数 --------------------
def ensure_output_directory(date_str):
    out_dir = f"/home/kemove/project/DC/output/1_paper/{date_str}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _new_session():
    """带重试的 requests session"""
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "GET"],
    )
    sess = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

# -------------------- 关键：获取“当天”的条数 --------------------
def get_today_papers_count(recent_url: str) -> int:
    """
    读取 /recent 页面，找到匹配今天(或指定日)的 <h3> 标题，
    优先解析 '(showing ... entries)' 的“entries 前的最后一个数字”；
    若缺失则统计该标题之后第一个 <dl> 中 <dt> 的个数。
    """
    try:
        sess = _new_session()
        resp = sess.get(recent_url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # arXiv 日期标题形如：Thu, 4 Sep 2025（注意 day 无前导 0）
        dow = today_dt.strftime("%a")
        day = today_dt.day
        mon_year = today_dt.strftime("%b %Y")
        target_prefix = f"{dow}, {day} {mon_year}"
        print(f"Looking for date header: {target_prefix}")

        for h3 in soup.find_all("h3"):
            # 规范空白，避免 \xa0 / 多空格影响匹配
            header_norm = " ".join(h3.get_text(" ", strip=True).split())
            if not header_norm.startswith(target_prefix):
                continue

            print(f"Found header: {header_norm}")

            # 直接抓 entries 前的最后一个数字（最稳）
            m = re.search(r"showing.*?(\d+)\s+entries", header_norm, flags=re.I)
            if m:
                total = int(m.group(1))
                return total

            # 兜底：结构解析，找“之后第一个 <dl>”
            dl = h3.find_next("dl")
            if dl:
                return len(dl.find_all("dt"))

            return 0

        print("Date header not found on page.")
        return 0

    except requests.RequestException as e:
        print(f"Error getting today's paper count: {e}")
        return 0



# -------------------- 翻页抓取 --------------------
def fetch_papers_from_recent(recent_url: str, category: str, total_papers: int):
    """
    通过 /recent?skip=&show= 翻页抓取，直到收集到 total_papers 篇为止
    （recent 列表是按时间倒序的全局列表，我们只取前 N 篇，即当天的 N 篇）。
    """
    collected = 0
    papers = []
    papers_per_page = 50
    sess = _new_session()

    while collected < total_papers:
        skip = collected
        page_url = f"{recent_url}?skip={skip}&show={papers_per_page}"
        print(f"[{category}] Fetch page: {page_url}")

        try:
            r = sess.get(page_url, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")

            dl = soup.find("dl")
            if not dl:
                print("No <dl> block found on this page.")
                break

            dt_elements = dl.find_all("dt")
            dd_elements = dl.find_all("dd")
            if not dt_elements:
                print("No <dt> entries on this page.")
                break

            for dt, dd in zip(dt_elements, dd_elements):
                if collected >= total_papers:
                    break

                try:
                    # 论文详情页（abstract）
                    abs_rel = dt.find("a", title="Abstract")["href"]
                    abs_url = base_url + abs_rel
                    pr = sess.get(abs_url, timeout=10)
                    pr.raise_for_status()
                    psoup = BeautifulSoup(pr.content, "html.parser")

                    # HTML 或 PDF 链接
                    html_link = dt.find("a", title="View HTML")
                    if html_link:
                        html_url = html_link["href"]
                    else:
                        pdf_rel = dt.find("a", title="Download PDF")["href"]
                        html_url = base_url + pdf_rel

                    # 标题 / 摘要 / 作者
                    title = (
                        psoup.find("h1", class_="title mathjax")
                        .text.strip()
                        .replace("Title:", "")
                        .strip()
                    )
                    abstract = (
                        psoup.find("blockquote", class_="abstract mathjax")
                        .text.strip()
                        .replace("Abstract:", "")
                        .strip()
                    )
                    authors = (
                        psoup.find("div", class_="authors")
                        .text.strip()
                        .replace("Authors:", "")
                        .strip()
                    )

                    # 提交日期
                    dateline = psoup.find("div", class_="dateline").text.strip()
                    m = re.search(r"Submitted on (\d+ \w+ \d+)", dateline)
                    date = m.group(1) if m else "N/A"

                    # DOI
                    doi_element = psoup.find("td", class_="tablecell arxivdoi")
                    doi = (
                        doi_element.find("a")["href"]
                        if doi_element and doi_element.find("a")
                        else "N/A"
                    )

                    papers.append(
                        {
                            "count": len(papers),
                            "url": html_url,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "date": date,
                            "doi": doi,
                            "category": category,
                            "Text that mentions Dataset": None,
                            "Dataset Links": None,
                        }
                    )
                    collected += 1

                    print(f"[{category}] {collected}/{total_papers}: {title}")
                    time.sleep(0.3)  # 温和点

                except Exception as e:
                    print(f"Error fetching a paper detail: {e}")
                    continue

            # 页面间稍作等待
            time.sleep(1.0)

        except requests.RequestException as e:
            print(f"Request failed for page {page_url}: {e}")
            break

    return papers

# -------------------- 入口流程 --------------------
def dataset_crawl():
    print(f"Starting paper collection for {today}")
    out_dir = ensure_output_directory(today)

    for category in categories:
        print(f"\nProcessing category: {category}")
        recent_url = f"{base_url}/list/{category}/recent"

        # 1) 找到当天条数
        n_today = get_today_papers_count(recent_url)
        print(f"[{category}] today entries: {n_today}")
        if n_today == 0:
            print(f"No papers found in {category} for {today}")
            continue

        # 2) 翻页抓满 N 篇
        papers = fetch_papers_from_recent(recent_url, category, n_today)

        # 3) 保存
        out_file = os.path.join(out_dir, f"{category}.json")
        if papers:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(papers)} papers to {out_file}")
        else:
            print(f"No papers were successfully collected for {category}")

if __name__ == "__main__":
    dataset_crawl()
