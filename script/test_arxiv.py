#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone arXiv API probe:
- 直连 export.arxiv.org/api/query（HTTP/HTTPS 可切换）
- 可调：分类、最近天数、页大小、页数、请求间隔、超时
- 打印：每次请求的 URL、状态码、是否 429、Retry-After、解析到的 entry 数、最早/最晚 updated
- 自带指数退避（针对 429 与超时），并尊重 Retry-After
- 仅依赖 requests、xml.etree.ElementTree（标准库），不依赖 feedparser
"""

import argparse
import random
import time
import datetime as dt
from typing import Tuple, List, Optional
import requests
import xml.etree.ElementTree as ET

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
API_PATH = "/api/query"

def utc_now():
    return dt.datetime.utcnow()

def fmt_arxiv_time(t: dt.datetime) -> str:
    # arXiv api supports lastUpdatedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
    return t.strftime("%Y%m%d%H%M")

def build_params(category: str, days: int, start: int, max_results: int) -> dict:
    now = utc_now()
    begin = now - dt.timedelta(days=days)
    q = f"cat:{category} AND lastUpdatedDate:[{fmt_arxiv_time(begin)} TO {fmt_arxiv_time(now)}]"
    return {
        "search_query": q,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": str(start),
        "max_results": str(max_results),
    }

def parse_atom_entries(xml_text: str) -> Tuple[int, Optional[str], Optional[str]]:
    """
    返回: (entry_count, newest_updated_iso, oldest_updated_iso)
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0, None, None
    entries = root.findall("atom:entry", ATOM_NS)
    times: List[dt.datetime] = []
    for e in entries:
        u = e.findtext("atom:updated", default="", namespaces=ATOM_NS)
        if not u:
            continue
        try:
            # Atom updated 示例: 2025-11-11T23:59:59Z
            t = dt.datetime.strptime(u, "%Y-%m-%dT%H:%M:%SZ")
            times.append(t)
        except Exception:
            pass
    if not entries:
        return 0, None, None
    if times:
        newest = max(times).isoformat() + "Z"
        oldest = min(times).isoformat() + "Z"
    else:
        newest = oldest = None
    return len(entries), newest, oldest

def jittered_sleep(base_sec: float):
    time.sleep(base_sec + random.uniform(0, base_sec * 0.3))

def one_request(
    session: requests.Session,
    base_url: str,
    params: dict,
    timeout_conn: float,
    timeout_read: float,
    min_interval: float,
    attempt: int,
) -> requests.Response:
    # 统一节流（请求之间至少 min_interval 秒）
    jittered_sleep(min_interval)
    t0 = time.time()
    resp = session.get(
        base_url + API_PATH,
        params=params,
        timeout=(timeout_conn, timeout_read),
    )
    dt_ms = int((time.time() - t0) * 1000)
    print(f"[HTTP] {resp.status_code} {resp.url}  {dt_ms}ms  len={resp.headers.get('Content-Length','?')}")
    return resp

def main():
    ap = argparse.ArgumentParser(description="Probe arXiv export API")
    ap.add_argument("--category", default="cs.AI", help="arXiv category, e.g. cs.AI/cs.CL/cs.CV")
    ap.add_argument("--days", type=int, default=7, help="lookback days for lastUpdatedDate")
    ap.add_argument("--max-results", type=int, default=200, help="max_results per page (50~200建议)")
    ap.add_argument("--pages", type=int, default=1, help="how many pages to fetch")
    ap.add_argument("--protocol", choices=["http", "https"], default="https", help="use http or https")
    ap.add_argument("--timeout-connect", type=float, default=10.0, help="connect timeout seconds")
    ap.add_argument("--timeout-read", type=float, default=40.0, help="read timeout seconds")
    ap.add_argument("--min-interval", type=float, default=3.0, help="min seconds between requests (rate limit)")
    ap.add_argument("--user-agent", default="dataset-miner-probe/1.0 (+mailto:you@example.com)",
                    help="custom UA per arXiv best practices")
    ap.add_argument("--respect-retry-after", action="store_true", help="sleep by Retry-After on 429/503 if provided")
    ap.add_argument("--backoff-base", type=float, default=8.0, help="base backoff seconds on 429/timeouts")
    args = ap.parse_args()

    base_url = f"{args.protocol}://export.arxiv.org"

    s = requests.Session()
    s.headers.update({"User-Agent": args.user_agent})

    start = 0
    for page_idx in range(args.pages):
        params = build_params(args.category, args.days, start, args.max_results)
        attempt = 0
        backoff = args.backoff_base
        while True:
            try:
                resp = one_request(
                    s, base_url, params,
                    timeout_conn=args.timeout_connect,
                    timeout_read=args.timeout_read,
                    min_interval=args.min_interval,
                    attempt=attempt,
                )
                # 429/5xx 处理
                if resp.status_code in (429, 500, 502, 503, 504):
                    ra = resp.headers.get("Retry-After")
                    if args.respect-retry-after and ra:
                        try:
                            sleep_s = float(ra)
                        except ValueError:
                            # 有些服务会给日期字符串，简单回退到 backoff
                            sleep_s = backoff
                    else:
                        sleep_s = backoff
                    print(f"[WARN] status={resp.status_code}; sleeping {sleep_s:.1f}s then retry…")
                    time.sleep(sleep_s)
                    attempt += 1
                    # 指数退避(上限60s)
                    backoff = min(backoff * 1.7, 60.0)
                    continue

                # 正常 or 4xx/其他
                if resp.status_code != 200:
                    print(f"[ERROR] unexpected status={resp.status_code}, body[:200]={resp.text[:200]!r}")
                    break

                # 解析条目
                n, newest, oldest = parse_atom_entries(resp.text)
                print(f"[OK] page={page_idx} entries={n} newest={newest} oldest={oldest}")
                if n == 0:
                    print("[INFO] empty page -> stop paging")
                    break

                # 下一页
                start += args.max_results
                break

            except requests.exceptions.ReadTimeout:
                print(f"[WARN] ReadTimeout; backoff {backoff:.1f}s then retry…")
                time.sleep(backoff)
                attempt += 1
                backoff = min(backoff * 1.7, 60.0)
            except requests.exceptions.ConnectTimeout:
                print(f"[WARN] ConnectTimeout; backoff {backoff:.1f}s then retry…")
                time.sleep(backoff)
                attempt += 1
                backoff = min(backoff * 1.7, 60.0)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] RequestException: {e}")
                break

    print("\nDone.")

if __name__ == "__main__":
    main()
