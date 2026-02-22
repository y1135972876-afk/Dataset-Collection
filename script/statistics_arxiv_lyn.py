import requests
import xml.etree.ElementTree as ET

BASE_URL = "http://export.arxiv.org/api/query"

# 你原来的类别
CATEGORIES = [
    "cs.IR",
    "cs.DB",
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.MA",
]

YEARS = ["2024", "2025"]

# 建议：UA 里写清楚用途 + 联系方式，符合 arxiv 要求
session = requests.Session()
session.headers.update({
    "User-Agent": "kzlab-arxiv-stats/0.1 (mailto:your_email@example.com)"
})


from typing import Optional

def get_count_for_category_year(category: str, year: str) -> Optional[int]:
    """
    使用 arXiv API，统计某一类别在某一年提交(submitted)的文章总数。
    """
    start = f"{year}01010000"   # 该年 1 月 1 日 00:00 (GMT)
    end   = f"{year}12312359"   # 该年 12 月 31 日 23:59 (GMT)

    search_query = (
        f"cat:{category} AND submittedDate:[{start} TO {end}]"
    )

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": 1,   # 只为了拿 totalResults
        # 不是必须，但可以加上排序字段
        # "sortBy": "submittedDate",
        # "sortOrder": "ascending",
    }

    try:
        resp = session.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed for {category} {year}: {e}")
        return None

    try:
        root = ET.fromstring(resp.text)
        ns = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
        total_elem = root.find("opensearch:totalResults", ns)
        if total_elem is None or not total_elem.text:
            print(f"Cannot find totalResults for {category} {year}")
            return None
        return int(total_elem.text)
    except Exception as e:
        print(f"Parse XML failed for {category} {year}: {e}")
        return None


if __name__ == "__main__":
    for year in YEARS:
        print(f"\n==== {year} 年 ====")
        total_year = 0

        for cat in CATEGORIES:
            cnt = get_count_for_category_year(cat, year)
            if cnt is None:
                print(f"{cat}: 获取失败")
                continue
            print(f"{cat}: {cnt} 篇")
            total_year += cnt

        print(f">>> {year} 年这几个类别总共：{total_year} 篇")
