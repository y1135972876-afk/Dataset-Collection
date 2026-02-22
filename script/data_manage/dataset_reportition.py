# -*- coding: utf-8 -*-
import json, random, os
from collections import defaultdict
from typing import Dict, List, Tuple

# =========================
# 1) 在这里填写你的路径/链接
# =========================
TRAIN_FILE_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/1212.json"
TEST_FILE_PATH  = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_dataset.json"

# 输出仍然是这两个文件（覆盖写回）
OUT_TRAIN_PATH  = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json"
OUT_TEST_PATH   = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json"

# 重配模式： "papers"（按论文数 1/12） 或 "sentences"（按句子数 1/12，paper 仍不拆分，近似）
MODE = "papers"
SEED = 42
TRIALS_FOR_SENTENCES = 80   # 只在 MODE="sentences" 时有效

# =========================
# 2) 工具函数
# =========================
def load_json(path_or_url: str) -> dict:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(path_or_url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return json.load(f)

def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def group_by_paper(data_obj: dict) -> Dict[str, List[dict]]:
    papers = defaultdict(list)
    for it in data_obj.get("data", []):
        papers[it["paper_name"]].append(it)
    return dict(papers)

def count_papers_and_sentences(papers_dict: Dict[str, List[dict]]) -> Tuple[int, int]:
    p = len(papers_dict)
    s = 0
    for plist in papers_dict.values():
        for item in plist:
            s += len(item.get("sentences", []))
    return p, s

def flatten_to_json(papers_dict: Dict[str, List[dict]]) -> dict:
    data_list = []
    for plist in papers_dict.values():
        data_list.extend(plist)  # 不改变 item 结构
    return {"data": data_list}

def print_stats(title: str, papers_dict: Dict[str, List[dict]]):
    p, s = count_papers_and_sentences(papers_dict)
    print(f"{title:<12} | papers={p:>6} | sentences={s:>8}")

# =========================
# 3) 两种重配策略
# =========================
def adjust_by_papers(train_papers: Dict[str, List[dict]],
                     test_papers: Dict[str, List[dict]],
                     seed: int = 42):
    """
    目标：整体达到 训练:测试:验证 = 10:1:1。
    这里先只“定住测试 ≈ 总 paper 的 1/12”，验证集之后由训练脚本对 1212.json 再切 1/12。
    返回：new_train, new_test, total_papers, target_test, target_val
    """
    rng = random.Random(seed)
    total_papers = len(train_papers) + len(test_papers)
    target_each = round(total_papers / 12.0)  # 测试与验证各 ~1/12
    target_test = target_each
    target_val  = target_each

    cur_test = len(test_papers)
    if cur_test < target_test:
        need = target_test - cur_test
        pool = list(train_papers.keys())
        rng.shuffle(pool)
        for name in pool[:need]:
            test_papers[name] = train_papers.pop(name)
    elif cur_test > target_test:
        reduce = cur_test - target_test
        pool = list(test_papers.keys())
        rng.shuffle(pool)
        for name in pool[:reduce]:
            train_papers[name] = test_papers.pop(name)

    return train_papers, test_papers, total_papers, target_test, target_val

def adjust_by_sentences(train_papers: Dict[str, List[dict]],
                        test_papers: Dict[str, List[dict]],
                        seed: int = 42,
                        trials: int = 50):
    """
    让测试集“句子数”更贴近总句子数的 1/12（paper 仍整体迁移，不拆分，避免信息泄漏）。
    简单启发式：多次尝试在两侧移动较大的 paper，使测试句子数贴近目标。
    """
    rng = random.Random(seed)

    def sent_of(pd): 
        return count_papers_and_sentences(pd)[1]

    total_sent   = sent_of(train_papers) + sent_of(test_papers)
    target_test_sent = round(total_sent / 12.0)

    # 初始化
    best_train, best_test = dict(train_papers), dict(test_papers)
    best_gap = abs(sent_of(test_papers) - target_test_sent)

    def paper_sent_list(pd):
        res = []
        for name, plist in pd.items():
            s = sum(len(it.get("sentences", [])) for it in plist)
            res.append((name, s))
        return res

    for _ in range(trials):
        cur_train = dict(train_papers)
        cur_test  = dict(test_papers)
        from_train = paper_sent_list(cur_train)
        from_test  = paper_sent_list(cur_test)
        rng.shuffle(from_train)
        rng.shuffle(from_test)

        # 贪心小步微调几次
        for _inner in range(3):
            cur = sent_of(cur_test)
            gap = target_test_sent - cur
            if gap > 0:
                # 需要增加 test：从 train 搬一个大的过去
                from_train.sort(key=lambda x: x[1], reverse=True)
                moved = False
                for name, _s in from_train:
                    if name in cur_train:
                        cur_test[name] = cur_train.pop(name)
                        moved = True
                        break
                if not moved: break
            elif gap < 0:
                # 需要减少 test：从 test 搬一个大的回来
                from_test.sort(key=lambda x: x[1], reverse=True)
                moved = False
                for name, _s in from_test:
                    if name in cur_test:
                        cur_train[name] = cur_test.pop(name)
                        moved = True
                        break
                if not moved: break
            else:
                break

        gap_now = abs(sent_of(cur_test) - target_test_sent)
        if gap_now < best_gap:
            best_gap, best_train, best_test = gap_now, cur_train, cur_test

    # 纸面上的“目标 paper 数”（用于后续验证集 paper 粒度估算）
    new_total_papers = len(best_train) + len(best_test)
    target_each = round(new_total_papers / 12.0)
    return best_train, best_test, new_total_papers, target_each, target_each

# =========================
# 4) “一键运行”主流程（非 CLI）
# =========================
def main():
    # 读取
    train_obj = load_json(TRAIN_FILE_PATH)
    test_obj  = load_json(TEST_FILE_PATH)
    train_papers = group_by_paper(train_obj)
    test_papers  = group_by_paper(test_obj)

    # 原始统计
    print("="*72)
    print("原始统计")
    print("="*72)
    print_stats("TRAIN+VAL", train_papers)
    print_stats("TEST",      test_papers)
    tp0, ts0 = count_papers_and_sentences(train_papers)
    up0, us0 = count_papers_and_sentences(test_papers)
    print(f"{'TOTAL':<12} | papers={tp0+up0:>6} | sentences={ts0+us0:>8}")

    # 调整
    if MODE == "papers":
        new_train, new_test, total_papers, target_test, target_val = adjust_by_papers(
            train_papers, test_papers, seed=SEED
        )
        print("\n按 paper 数重配：目标 test≈总paper的 1/12，验证集稍后从训练文件再切 1/12。")
    elif MODE == "sentences":
        new_train, new_test, total_papers, target_test, target_val = adjust_by_sentences(
            train_papers, test_papers, seed=SEED, trials=TRIALS_FOR_SENTENCES
        )
        print("\n按句子数重配（paper 不拆分）：使 test 的句子数更接近总句子数的 1/12，验证集稍后再切 1/12。")
    else:
        raise ValueError("MODE 只能是 'papers' 或 'sentences'")

    # 保存（覆盖/或另存）
    save_json(flatten_to_json(new_train), OUT_TRAIN_PATH)
    save_json(flatten_to_json(new_test),  OUT_TEST_PATH)

    # 调整后统计
    print("\n" + "="*72)
    print("调整后统计（已写回到输出路径）")
    print("="*72)
    print_stats("TRAIN+VAL", new_train)
    print_stats("TEST",      new_test)
    tp1, ts1 = count_papers_and_sentences(new_train)
    up1, us1 = count_papers_and_sentences(new_test)
    print(f"{'TOTAL':<12} | papers={tp1+up1:>6} | sentences={ts1+us1:>8}")

    # 给出 train_test_split 的建议 test_size（让验证集 ≈ 总体 1/12，paper 口径）
    desired_val_papers = round((tp1 + up1) / 12.0) if (tp1 + up1) > 0 else 0
    suggested_test_size = (desired_val_papers / tp1) if tp1 > 0 else 0.0
    print("\n" + "="*72)
    print("train_test_split 建议")
    print("="*72)
    print(f"建议：test_size = {suggested_test_size:.6f}  (random_state=42)")
    print("含义：在训练脚本中对 1212.json 再切 validation，使验证集 ≈ 总体 1/12；测试集已固定为 ≈ 1/12。")
    print("="*72)

if __name__ == "__main__":
    main()
