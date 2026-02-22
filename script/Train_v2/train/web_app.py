#!/usr/bin/env python3
"""Savvy Dataset Crawler Web 控制台 + 检索模型接入"""

import os
import glob
import json
import threading
import time
from typing import List, Dict

from flask import Flask, render_template, request, jsonify

# ================== 1. 导入爬虫主循环 ==================
# 这份脚本里需要有：CONFIG, main_loop, setup_logging, install_perf_probes
from run_loop_bert_label import CONFIG, main_loop, setup_logging, install_perf_probes  # type: ignore

# ================== 2. 导入检索框架 ==================
import sys
from pathlib import Path

app = Flask(__name__)

# web_app.py 所在目录：.../script/Train_v2/train
WEB_DIR = Path(__file__).resolve().parent
# script 目录：.../script
SCRIPT_DIR = WEB_DIR.parents[1]
# dataset_retrieval 目录：.../script/data_manage/dataset_retrieval
DATASET_RETRIEVAL_DIR = SCRIPT_DIR / "dataset_retrieval"

sys.path.insert(0, str(DATASET_RETRIEVAL_DIR))
print("[DEBUG] DATASET_RETRIEVAL_DIR =", DATASET_RETRIEVAL_DIR)

try:
    from retrieval_framework import RetrievalFramework  # 文件名需是 retrieval_framework.py
except ImportError as e:
    print("[ERROR] cannot import RetrievalFramework:", e)
    RetrievalFramework = None  # type: ignore
    
    

# ======================================================================
#                          一、数据索引（关键词检索）
# ======================================================================

DATA_INDEX: List[Dict] = []


def load_index() -> List[Dict]:
    """
    从 CONFIG.OUT_FINAL 目录加载所有 *.json 结果文件。
    每个文件是一组记录：
    {
      "Paper Link": "...",
      "Dataset Link": "...",
      "Dataset Description": "...",
      "Timestamp": "..."
    }
    """
    out_dir = CONFIG.OUT_FINAL
    os.makedirs(out_dir, exist_ok=True)

    records: List[Dict] = []
    for fp in glob.glob(os.path.join(out_dir, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, list):
                records.extend(items)
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")
            continue

    # 去重（Paper Link + Dataset Link）
    seen = set()
    dedup: List[Dict] = []
    for r in records:
        key = (r.get("Paper Link", "").strip(), r.get("Dataset Link", "").strip())
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    print(f"[INFO] loaded {len(dedup)} records from {out_dir}")
    return dedup


def search_index(query: str, limit: int = 50) -> List[Dict]:
    """
    简单全文检索：
    - 把 query 按空格分词
    - 在 Paper Link / Dataset Link / Dataset Description 上打分
    """
    if not DATA_INDEX:
        return []

    q = (query or "").strip()
    if not q:
        # 没 query 时：按时间降序返回最新的若干条
        return sorted(
            DATA_INDEX,
            key=lambda x: x.get("Timestamp", ""),
            reverse=True,
        )[:limit]

    tokens = [t.lower() for t in q.split() if t.strip()]
    results = []
    for r in DATA_INDEX:
        text = " ".join(
            [
                r.get("Paper Link", ""),
                r.get("Dataset Link", ""),
                r.get("Dataset Description", ""),
            ]
        ).lower()
        score = 0
        for t in tokens:
            if t in text:
                score += text.count(t)
        if score > 0:
            item = dict(r)
            item["_score"] = score
            results.append(item)

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return results[:limit]


# ======================================================================
#                          一点五、检索模型（GTE）
# ======================================================================

RETRIEVAL = None  # type: ignore
RETRIEVAL_READY = False


def init_retrieval_model():
    """
    在 Flask 进程启动时初始化检索模型。
    只加载一次，后续请求复用。
    """
    global RETRIEVAL, RETRIEVAL_READY

    if RetrievalFramework is None:
        print("[Retrieval] RetrievalFramework 未导入，跳过初始化")
        return

    # 路径用你在 CLI main.py 里用的那两个
    model_path = "/home/kzlab/muse/Savvy/Data_collection/script/dataset_retrieval/models/Alibaba-gte-large-en-v1.5"
    data_dir = "/home/kzlab/muse/Savvy/Data_collection/script/dataset_retrieval/datasets"

    try:
        print("[Retrieval] 初始化 RetrievalFramework ...")
        framework = RetrievalFramework(data_dir=data_dir)
        framework.initialize_retriever("gte", model_name=model_path)
        # 如果你上一步在 RetrievalFramework 里实现了 search()，这里就可以用了
        RETRIEVAL = framework
        RETRIEVAL_READY = True
        stats = getattr(framework, "get_data_statistics", lambda: None)()
        if stats:
            print(f"[Retrieval] 初始化完成。queries={stats.get('num_queries')}, datasets={stats.get('num_datasets')}")
        else:
            print("[Retrieval] 初始化完成。")
    except Exception as e:
        print(f"[ERROR] 初始化检索模型失败: {e}")
        RETRIEVAL = None
        RETRIEVAL_READY = False


# ======================================================================
#                          二、爬取后台线程
# ======================================================================

_crawl_thread = None  # type: ignore
_crawl_stop_event = threading.Event()
_crawl_lock = threading.Lock()
_crawl_status = {
    "running": False,
    "last_start": None,  # type: ignore
    "last_stop": None,   # type: ignore
}


def _crawl_entry():
    """
    在后台线程里运行 main_loop，支持 stop_event。
    需要你的 main_loop(stop_event=None) 支持 stop_event 参数。
    """
    setup_logging()
    install_perf_probes()
    try:
        main_loop(stop_event=_crawl_stop_event)
    finally:
        with _crawl_lock:
            _crawl_status["running"] = False
            _crawl_status["last_stop"] = time.time()


# ======================================================================
#                          三、配置相关接口
# ======================================================================

def _config_snapshot() -> Dict:
    """把关键 CONFIG 项导出给前端。"""
    return {
        "DESC_SOURCE": getattr(CONFIG, "DESC_SOURCE", "llm"),
        "LINK_CLASSIFIER": getattr(CONFIG, "LINK_CLASSIFIER", "rule"),
        "LINK_PICKER": getattr(CONFIG, "LINK_PICKER", "hybrid"),
        "LATEX_LLM_DISABLED": bool(getattr(CONFIG, "LATEX_LLM_DISABLED", False)),
        "LAST_N_DAYS": int(getattr(CONFIG, "LAST_N_DAYS", 7)),
        "BERT_THRESHOLD": float(getattr(CONFIG, "BERT_THRESHOLD", 0.5)),
        "WORKERS": int(getattr(CONFIG, "WORKERS", 1)),
        "PDF_MAX_CONCURRENCY": int(getattr(CONFIG, "PDF_MAX_CONCURRENCY", 4)),
        "PRIMARY_HOST_TOPK": int(getattr(CONFIG, "PRIMARY_HOST_TOPK", 5)),
    }


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    # GET：读配置
    if request.method == "GET":
        return jsonify(_config_snapshot())

    # POST：写配置（只允许改部分字段，避免踩其他东西）
    data = request.get_json() or {}

    # 字符字段
    for key in ["DESC_SOURCE", "LINK_CLASSIFIER", "LINK_PICKER"]:
        if key in data and hasattr(CONFIG, key):
            setattr(CONFIG, key, str(data[key]).lower())

    # 数值字段
    def _set_int(k: str):
        if k in data and hasattr(CONFIG, k):
            try:
                setattr(CONFIG, k, int(data[k]))
            except Exception:
                pass

    def _set_float(k: str):
        if k in data and hasattr(CONFIG, k):
            try:
                setattr(CONFIG, k, float(data[k]))
            except Exception:
                pass

    _set_int("LAST_N_DAYS")
    _set_float("BERT_THRESHOLD")
    _set_int("WORKERS")
    _set_int("PDF_MAX_CONCURRENCY")
    _set_int("PRIMARY_HOST_TOPK")

    # 布尔字段
    if "LATEX_LLM_DISABLED" in data and hasattr(CONFIG, "LATEX_LLM_DISABLED"):
        setattr(CONFIG, "LATEX_LLM_DISABLED", bool(data["LATEX_LLM_DISABLED"]))

    return jsonify({"ok": True, "config": _config_snapshot()})


# ======================================================================
#                          四、爬取控制接口
# ======================================================================

@app.route("/api/crawl/start", methods=["POST"])
def api_crawl_start():
    global _crawl_thread, _crawl_stop_event
    with _crawl_lock:
        if _crawl_status["running"]:
            return jsonify({"ok": True, "status": "running"})

        _crawl_stop_event = threading.Event()
        _crawl_thread = threading.Thread(target=_crawl_entry, daemon=True)
        _crawl_status["running"] = True
        _crawl_status["last_start"] = time.time()
        _crawl_thread.start()
        return jsonify({"ok": True, "status": "started"})


@app.route("/api/crawl/stop", methods=["POST"])
def api_crawl_stop():
    global _crawl_stop_event
    with _crawl_lock:
        if not _crawl_status["running"]:
            return jsonify({"ok": True, "status": "idle"})
        _crawl_stop_event.set()
        return jsonify({"ok": True, "status": "stopping"})


@app.route("/api/crawl/status", methods=["GET"])
def api_crawl_status():
    with _crawl_lock:
        running = _crawl_status["running"]
        return jsonify(
            {
                "running": running,
                "status_text": "运行中" if running else "已停止",
                "last_start": _crawl_status["last_start"],
                "last_stop": _crawl_status["last_stop"],
            }
        )


# ======================================================================
#                          五、检索接口
# ======================================================================

@app.route("/api/search")
def api_search():
    """
    检索接口：
    - 无 q：返回按时间倒序的最近 top_k 条（不带 score）
    - 有 q：只使用 RetrievalFramework.search（GTE 模型）
    """
    q = request.args.get("q", "", type=str).strip()

    # ========= 1. 没有查询词：默认展示最新 50 条 =========
    if not q:
        top_k_default = request.args.get("top_k", 50, type=int)

        results = sorted(
            DATA_INDEX,
            key=lambda x: x.get("Timestamp", ""),
            reverse=True,
        )[:top_k_default]

        formatted = [
            {
                "paper_link": r.get("Paper Link", ""),
                "dataset_link": r.get("Dataset Link", ""),
                "description": r.get("Dataset Description", ""),
                "timestamp": r.get("Timestamp", ""),
                # 注意：这里故意不带 "score"
            }
            for r in results
        ]
        return jsonify({"results": formatted, "mode": "default"})

    # ========= 2. 有查询词：语义检索 =========
    # 如果模型没初始化，保持你原来的 500 返回
    if not (RETRIEVAL_READY and RETRIEVAL is not None):
        return jsonify({
            "results": [],
            "mode": "semantic",
            "error": "retrieval model not initialized"
        }), 500

    top_k = request.args.get("top_k", 10, type=int)
    with_instruction = request.args.get("with_instruction", "0") in ["1", "true", "True"]

    try:
        # 这里 search 返回的结构已经是 list[dict]，字段名和前端一致
        results = RETRIEVAL.search(
            query=q,
            top_k=top_k,
            with_instruction=with_instruction,
        )
        return jsonify({"results": results, "mode": "semantic"})
    except Exception as e:
        print(f"[ERROR] semantic search failed: {e}")
        return jsonify({
            "results": [],
            "mode": "semantic",
            "error": "semantic search failed"
        }), 500




@app.route("/api/stats")
def api_stats():
    return jsonify({"count": len(DATA_INDEX)})


# ======================================================================
#                          六、页面路由
# ======================================================================

@app.route("/")
def index():
    return render_template("index.html")


def main():
    global DATA_INDEX
    DATA_INDEX = load_index()

    # 初始化检索模型（只在进程启动时执行一次）
    init_retrieval_model()

    # 注意：debug=True 会启用 reloader，进程会跑两份，模型会加载两遍。
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
