import json

# 假设你的数据保存在 data.json 文件中
with open("/home/kzlab/muse/Savvy/Data_collection/dataset/dataset-descrption/dataset/json/1212.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 统计 paper 数量
num_papers = len(dataset["data"])

# 统计句子数量
num_sentences = sum(len(item["sentences"]) for item in dataset["data"])

print(f"总共有 {num_papers} 篇 paper")
print(f"总共有 {num_sentences} 个句子")
