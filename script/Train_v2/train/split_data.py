from collections import OrderedDict, Counter
import copy, hashlib
from typing import List, Tuple, Dict, Any
import json, os

# ---------------- helpers ----------------
def _safe_sent_count(item: Dict[str, Any]) -> int:
    sents = item.get("sentences", [])
    return len(sents) if isinstance(sents, list) else 0

def _group_by_paper_preserve_order(items: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """按出现顺序分组，每个 paper 的条目保持原顺序；paper_name 缺失则生成唯一占位名。"""
    groups = OrderedDict()
    unknown_id = 0
    for it in items:
        pn = it.get("paper_name")
        if not isinstance(pn, str) or not pn.strip():
            pn = f"__UNKNOWN__:{unknown_id}"
            unknown_id += 1
        groups.setdefault(pn, []).append(it)
    return groups

def _paper_sentence_size(groups: "OrderedDict[str, List[Dict[str, Any]]]", paper_name: str) -> int:
    return sum(_safe_sent_count(x) for x in groups[paper_name])

# ---------------- two strategies ----------------
def _plan_equal_papers(groups: "OrderedDict[str, List[Dict[str, Any]]]", k: int) -> List[List[str]]:
    """按 paper 数等量分组（不打乱原顺序、不拆 paper）。"""
    papers = list(groups.keys())
    n = len(papers)
    k = max(1, min(k, n))
    per = (n + k - 1) // k
    return [papers[i*per : min((i+1)*per, n)] for i in range(k)]

def _plan_balance_sentences(groups: "OrderedDict[str, List[Dict[str, Any]]]", k: int) -> List[List[str]]:
    """按句子数贪心均衡（不拆 paper，尽量让每份句子数接近）。"""
    papers = list(groups.keys())
    n = len(papers)
    k = max(1, min(k, n))
    sized = [(p, _paper_sentence_size(groups, p)) for p in papers]
    sized.sort(key=lambda x: x[1], reverse=True)
    shards, loads = [[] for _ in range(k)], [0]*k
    for p, sz in sized:
        i = min(range(k), key=lambda j: loads[j])
        shards[i].append(p)
        loads[i] += sz
    return shards

# ---------------- public API (with switch) ----------------
def split_dataset(data: Dict[str, Any], k: int, mode: str = "papers") -> List[Dict[str, Any]]:
    """
    将数据按 paper 切成 k 份（若 paper 数 < k，则份数 <= paper 数），保证：
      - 不拆 paper；paper 内顺序不变；不改任何字段
      - 无随机性（稳定可复现）
    mode:
      - "papers"    : 按 paper 数等量分
      - "sentences" : 按句子数贪心均衡
    返回：列表，每个元素形如 { "data": [...] }
    """
    assert isinstance(data, dict) and isinstance(data.get("data"), list), "bad input JSON"
    groups = _group_by_paper_preserve_order(data["data"])
    if mode == "papers":
        plan = _plan_equal_papers(groups, k)
    elif mode == "sentences":
        plan = _plan_balance_sentences(groups, k)
    else:
        raise ValueError("mode must be 'papers' or 'sentences'")

    shards: List[Dict[str, Any]] = []
    for paper_list in plan:
        bucket = []
        for pn in paper_list:
            bucket.extend(groups[pn])
        shards.append({"data": copy.deepcopy(bucket)})  # 深拷贝，避免外部修改影响原数据
    return shards

# ---------------- optional: index & summary ----------------
def build_index_and_summary(shards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    生成索引与统计（不修改数据）：
    - index: 每份中的 paper 列表及其段落/句子数
    - summary: 全局与分片汇总
    """
    def _group(sh): return _group_by_paper_preserve_order(sh.get("data", []))
    index: List[Dict[str, Any]] = []
    summary = {"n_shards": len(shards), "shards": [], "total_papers": 0,
               "total_paragraphs": 0, "total_sentences": 0}
    for i, sh in enumerate(shards, 1):
        g = _group(sh)
        papers = list(g.keys())
        summary["total_papers"] += len(papers)
        p_para = sum(len(g[pn]) for pn in papers)
        p_sent = sum(_paper_sentence_size(g, pn) for pn in papers)
        for pn in papers:
            index.append({
                "shard": i,
                "paper_name": pn,
                "paragraphs": len(g[pn]),
                "sentences": _paper_sentence_size(g, pn)
            })
        summary["shards"].append({"shard": i, "papers": len(papers),
                                  "paragraphs": p_para, "sentences": p_sent})
        summary["total_paragraphs"] += p_para
        summary["total_sentences"] += p_sent
    return index, summary

# ---------------- strict lossless check ----------------
def _fingerprints(dataset: Dict[str, Any]) -> Counter:
    """
    用 (paper_name, paragraph_id, sent_idx, sha1(text)) 作为键做多重计数指纹。
    既校集合也校重数，能发现丢条/重复/改写/顺序错乱引起的计数不一致。
    """
    fp = Counter()
    for it in dataset.get("data", []):
        pn = it.get("paper_name")
        pid = it.get("paragraph_id")
        sents = it.get("sentences", [])
        if not isinstance(sents, list):
            continue
        for i, s in enumerate(sents):
            txt = s.get("text", "")
            if not isinstance(txt, str):
                txt = str(txt)
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            fp[(pn, pid, i, h)] += 1
    return fp

def assert_lossless(original: Dict[str, Any], shards: List[Dict[str, Any]]) -> None:
    """严格零损失校验：条目数、句子数、指纹计数三重一致，否则抛 AssertionError。"""
    ori_items = len(original.get("data", []))
    sh_items = sum(len(sh.get("data", [])) for sh in shards)
    assert ori_items == sh_items, f"items mismatch: {ori_items} vs {sh_items}"

    ori_sents = sum(len(x.get("sentences", []) or []) for x in original.get("data", []))
    sh_sents = sum(sum(len(x.get("sentences", []) or []) for x in sh.get("data", [])) for sh in shards)
    assert ori_sents == sh_sents, f"sentences mismatch: {ori_sents} vs {sh_sents}"

    fp_ori = _fingerprints(original)
    fp_sh = Counter()
    for sh in shards:
        fp_sh += _fingerprints(sh)
    assert fp_ori == fp_sh, "fingerprints mismatch (loss/duplication/rewrite detected)"
    
    
    
data = json.load(open("/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json", "r", encoding="utf-8"))
shards = split_dataset(data, k=20, mode="sentences")  # 或 "papers"
assert_lossless(data, shards)  # 不一致会抛错
for i, sh in enumerate(shards, 1):
    json.dump(sh, open(f"/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/split/train_part_{i:02d}.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
