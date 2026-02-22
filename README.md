# 代码部分介绍





### Dataset

我们爬取的部分dataset，人工打标签后用于训练和评测模型，已经上传到了hugging face（[hug2you/Dataset_Collection · Datasets at Hugging Face](https://huggingface.co/datasets/hug2you/Dataset_Collection)）。可以下载到/dataset文件夹下即可



### Models

有很多模型资料，可以从hugging_face获取，下载到/models文件夹下即可





基于bert-base-uncased（[google-bert/bert-base-uncased · Hugging Face](https://huggingface.co/google-bert/bert-base-uncased)）训练得到的

BERT-Gate 和 BERT-Desc均已上传至[hug2you (Yang Junzhe)](https://huggingface.co/hug2you)的BERT-Gate [hug2you/Bert-Gate · Hugging Face](https://huggingface.co/hug2you/Bert-Gate) 和 BERT-Desc  [hug2you/BERT-Desc · Hugging Face](https://huggingface.co/hug2you/BERT-Desc)

检索用 Embedding：Alibaba GTE-large 可从 [Alibaba-NLP/gte-large-en-v1.5 · Hugging Face](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) 下载

提取Datalink 的： DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 也可以从[RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 at main](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16/tree/main)得到

其他的：Longformer和 Bigbert 作为baseline，Longformer可以从





### 启动grobid

```
# 拉取 GROBID 镜像
docker pull lfoppiano/grobid:0.8.0

# 运行 GROBID 容器
docker run -t --rm -p 8071:8070 lfoppiano/grobid:0.8.0

```

判断API

```
# curl（Linux/macOS）
curl -s http://localhost:8071/api/isalive

```



