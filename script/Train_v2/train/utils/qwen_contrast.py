# -*- coding: utf-8 -*-
"""
用 Qwen API 重新打标签并与原始标签对比
输出: relabel_diff.jsonl （存放所有不一致样本）
"""

import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

# ========== 用户配置 ==========
DATA_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json"
OUT_PATH  = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/relabel_diff.jsonl"

# 请先设置环境变量 OPENAI_API_KEY 或直接在此处填写：
# os.environ["OPENAI_API_KEY"] = "你的Qwen或OpenAI兼容Key"
client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "qwen2.5-7b-instruct"  # 可以改成你本地/云端可用的模型
SLEEP_SEC = 1.2  # 请求间隔防止被限速

# ========== 核心函数 ==========

def ask_qwen_for_label(sentence: str) -> int:
    """调用 Qwen，让它判断该句子是否描述数据集"""
    prompt = f"""
请判断下面这句话是否是在描述一个“数据集”的内容。
如果是，请输出 1；如果不是，请输出 0。
仅返回数字，不要解释。

句子：
{sentence.strip()}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        text = response.choices[0].message.content.strip()
        if "1" in text:
            return 1
        elif "0" in text:
            return 0
        else:
            # 异常输出时 fallback
            return -1
    except Exception as e:
        print("❌ API Error:", e)
        return -1


def relabel_and_compare():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    diffs = []
    total_sent = 0
    diff_count = 0

    for paper in tqdm(data, desc="Processing papers"):
        for sent in paper.get("sentences", []):
            total_sent += 1
            text = sent.get("text", "")
            old_label = int(sent.get("label", 0))

            new_label = ask_qwen_for_label(text)
            time.sleep(SLEEP_SEC)

            if new_label not in (0, 1):
                continue

            if new_label != old_label:
                diff = {
                    "paper_name": paper.get("paper_name", ""),
                    "sentence": text,
                    "old_label": old_label,
                    "qwen_label": new_label,
                }
                diffs.append(diff)
                diff_count += 1

                # 立即写入一行，避免中途中断丢失
                with open(OUT_PATH, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(diff, ensure_ascii=False) + "\n")

    print(f"\n=== 任务完成 ===")
    print(f"共处理句子: {total_sent}")
    print(f"标签不一致的句子: {diff_count}")
    print(f"结果已保存至: {OUT_PATH}")


if __name__ == "__main__":
    relabel_and_compare()
