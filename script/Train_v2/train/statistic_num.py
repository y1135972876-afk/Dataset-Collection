# -*- coding: utf-8 -*-
"""
只统计 2025 年“疑似构建数据集”的论文数量：
- 仍然用 fetch_arxiv_recent + BertGate
- 但按页处理，遇到 2024 及以前就停，不再往后翻
- 捕获 RetryError，避免 export.arxiv.org 500 把脚本干崩
"""

from typing import Optional

from utils.utils_loop import CONFIG, BertGate, fetch_arxiv_recent
from requests.exceptions import RetryError

TARGET_YEAR = 2025


def parse_year(date_str: str) -> Optional[int]:
    """从 published / updated 中取年份"""
    if not date_str:
        return None
    s = str(date_str).strip()
    if len(s) < 4:
        return None
    try:
        return int(s[:4])
    except ValueError:
        return None


def main():
    # 这个天数只是用来让 fetch_arxiv_recent 的时间窗覆盖到 2025 全年，多一点无所谓
    CONFIG.LAST_N_DAYS = 400

    bert = BertGate()
    seen_ids = set()   # 跨所有类别全局去重（同一篇论文多个分类只算一次）
    total_2025 = 0

    print("CATEGORIES:", CONFIG.CATEGORIES)
    print("LAST_N_DAYS:", CONFIG.LAST_N_DAYS)
    print("BERT_THRESHOLD:", CONFIG.BERT_THRESHOLD)
    print("只统计年份 =", TARGET_YEAR)
    print("-" * 60)

    for cat in CONFIG.CATEGORIES:
        start = 0
        page_size = CONFIG.ARXIV_MAX_RESULTS
        cat_count = 0

        print(f"\n[{cat}] 开始拉取……")

        while True:
            # 保险：不要让 start 无限增大（碰到 arXiv 的 10000 限制）
            if start >= 10000:
                print(f"[{cat}] start={start} 已接近 arXiv 结果上限，停止这个类别")
                break

            try:
                page = fetch_arxiv_recent(cat, max_results=page_size, start=start)
            except RetryError as e:
                print(f"[{cat}] RetryError at start={start}: {e}，停止这个类别")
                break
            except Exception as e:
                print(f"[{cat}] 其它异常 at start={start}: {e}，停止这个类别")
                break

            if not page:
                # 没有更多结果了
                break

            # 标记：这一页是否已经见到 2024 或更早
            reached_before_target = False

            for e in page:
                aid = e.get("id")
                if not aid:
                    continue

                # 全局去重：同一篇 paper 只算一次
                if aid in seen_ids:
                    continue

                updated_year = parse_year(e.get("updated"))
                if updated_year is None:
                    continue
                if updated_year < TARGET_YEAR:
                    reached_before_target = True
                    break
                if updated_year > TARGET_YEAR:
                    continue

                # year == 2025，才是真正我们关心的
                seen_ids.add(aid)

                prob = float(bert.score(e["title"], e["abstract"]))
                if prob >= float(CONFIG.BERT_THRESHOLD):
                    cat_count += 1
                    total_2025 += 1

            if reached_before_target:
                print(f"[{cat}] 已经翻到 {TARGET_YEAR} 之前的年份，停止这个类别")
                break

            # 当前页条数不足 page_size，也说明没有更多了
            if len(page) < page_size:
                break

            start += page_size

        print(f"[{cat}] 2025 年通过 BERT 过滤的 dataset-like 论文数量 = {cat_count}")

    print("\n================================================")
    print(f"2025 年（所有类别合在一起，去重后）dataset-like 论文总数 = {total_2025}")
    print("================================================")


if __name__ == '__main__':
    main()


