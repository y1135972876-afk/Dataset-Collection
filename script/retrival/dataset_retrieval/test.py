#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用本地 Alibaba gte-large-en-v1.5 做一个全方位小体检：
1. 基础语义相似度（同义改写 vs 无关）
2. 否定与立场
3. 数字/边界条件敏感度
4. 小规模语义检索 demo
5. 语言边界（英文 vs 中文）

运行方式：
    python test_gte_large_en_v15.py
"""

import torch
from sentence_transformers import SentenceTransformer, util

# TODO: 改成你的本地模型路径
MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/retrival/dataset_retrieval/dataset_retrieval/models/Alibaba-gte-large-en-v1.5"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading model from: {MODEL_PATH}")
    print(f"💻 Device: {device}")

    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        trust_remote_code=True,   # ✅ 关键
    )
    _ = model.encode(["warmup"], normalize_embeddings=True)
    return model



def show_pair_similarity(model, s1, s2, desc=""):
    emb = model.encode([s1, s2], convert_to_tensor=True, normalize_embeddings=True)
    score = util.cos_sim(emb[0], emb[1]).item()

    print("\n" + "=" * 80)
    title = f"🧩 {desc}" if desc else "🧩 Pair similarity"
    print(title)
    print("-" * 80)
    print(f"句子1: {s1}")
    print(f"句子2: {s2}")
    print(f"✅ 余弦相似度: {score:.4f}")
    return score


def test_basic_similarity(model):
    print("\n\n🌟 [Test 1] 基础语义相似度：同义改写 vs 无关\n")

    s1 = "How can I reduce my AWS cloud costs?"
    paraphrase = "What are the best practices to optimize spending on Amazon Web Services?"
    unrelated = "The cat is sleeping on the sofa."

    show_pair_similarity(model, s1, paraphrase, "同义改写 / 语义相近（期望：相似度很高）")
    show_pair_similarity(model, s1, unrelated, "完全无关（期望：相似度明显更低）")


def test_negation_and_stance(model):
    print("\n\n🌟 [Test 2] 否定与立场：模型的“弱点”之一\n")

    pos = "I like this product. It works really well."
    neg = "I don't like this product. It works terribly."
    neutral = "This product is available in three different colors."

    show_pair_similarity(model, pos, neg, "正向 vs 反向（期望：其实它会给出“相对较高”的相似度）")
    show_pair_similarity(model, pos, neutral, "正向 vs 中性无关（期望：相似度比上面更低）")

    print("\n💡 观察点：")
    print("   - 如果正向 vs 反向 的相似度依然挺高，说明它主要看“在聊同一个东西”，")
    print("     对否定、情感极性不敏感，这在检索场景里是常见现象。")


def test_numbers_and_ranges(model):
    print("\n\n🌟 [Test 3] 数字与边界条件敏感度\n")

    s_base = "The tax rate for income over 200,000 dollars is 35%."
    s_close = "The tax rate for income over 180,000 dollars is 35%."
    s_diff = "The tax rate for income under 50,000 dollars is 10%."

    show_pair_similarity(model, s_base, s_close, "边界略有差异（200k vs 180k，期望：相似度很高）")
    show_pair_similarity(model, s_base, s_diff, "区间完全不同（>200k vs <50k，期望：仍然不低）")

    print("\n💡 观察点：")
    print("   - Embedding 模型对“数字/区间”的精确逻辑并不敏感，")
    print("     主要还是把它们当作“同一主题：税率说明”。")
    print("   - 在做严格规则判断（金额、日期、阈值）时，后面要交给 LLM 或专门逻辑处理。")


def test_mini_retrieval(model):
    print("\n\n🌟 [Test 4] 小规模语义检索 demo\n")

    corpus = [
        "This guide explains how to build an asynchronous proxy pool in Python using asyncio.",
        "Our refund policy allows customers to return items within 30 days of purchase.",
        "We describe how to fine-tune BERT models for text classification tasks.",
        "The company offers a flexible remote work policy for all full-time employees.",
        "This tutorial shows how to optimize MySQL queries for read-heavy workloads.",
    ]
    for i, c in enumerate(corpus):
        print(f"[Doc {i}] {c}")

    queries = [
        "How can I implement an async proxy manager in Python?",
        "What is your return and refund policy?",
        "Cheapest option for a read-heavy database workload?",
    ]

    # 先算文档向量
    doc_emb = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

    for q in queries:
        q_emb = model.encode([q], convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, doc_emb)[0]  # shape: [num_docs]

        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "=" * 80)
        print(f"🔎 Query: {q}")
        print("- Top 3 docs by semantic similarity:")
        for rank, (idx, score) in enumerate(ranked[:3], start=1):
            print(f"  #{rank} [Doc {idx}] (score={score:.4f}) -> {corpus[idx]}")

    print("\n💡 观察点：")
    print("   - 看看每个 Query 排名第一的是不是你直觉上最相关的那段。")
    print("   - 这基本就是你真实检索系统里“第一层召回”的效果预期。")


def test_language_boundary(model):
    print("\n\n🌟 [Test 5] 语言边界：英文模型对中文的表现\n")

    en = "How can I reduce my AWS cloud costs?"
    zh = "我如何降低在 AWS 上的云成本？"
    random_zh = "今天中午吃什么比较好？"

    show_pair_similarity(model, en, zh, "英文句子 vs 中文语义相同句子（期望：可能有点高，但不稳定）")
    show_pair_similarity(model, en, random_zh, "英文句子 vs 中文完全无关句子")

    print("\n💡 观察点：")
    print("   - gte-large-en-v1.5 是英文专用模型，对中文支持是“顺带”的，")
    print("     所以中文 query + 英文文档在你业务里最好只当作 bonus，不要太依赖。")


def main():
    model = load_model()

    test_basic_similarity(model)
    test_negation_and_stance(model)
    test_numbers_and_ranges(model)
    test_mini_retrieval(model)
    test_language_boundary(model)

    print("\n\n🎉 全部测试完成！你现在可以根据这些相似度和检索结果，")
    print("   大致心里有数：它在“同义改写 / 主题相关 / 否定 / 数字 / 中英文”这些维度上的表现如何。")


if __name__ == "__main__":
    main()
