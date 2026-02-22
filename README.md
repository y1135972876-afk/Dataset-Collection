# AutoDataset: Automatically Discovering and Collecting ML Datasets from Research Papers

This repository contains the implementation of the paper **“AutoDataset: An Automated System for Tracking and Collecting ML Datasets from Research Papers”**, including:

- an arXiv crawler and filter based on BERT-Gate
- extraction of dataset descriptions from PDFs with BERT-Desc
- dataset-link extraction from LaTeX sources (rules + optional LLM)
- dense retrieval with GTE and a lightweight web demo

> All figures and experimental results in the paper (e.g., growth curves of dataset papers over time, model comparisons, etc.) can be reproduced using scripts in this repository.

## Environment

- Python >= 3.9
- PyTorch + CUDA (optional but strongly recommended)
- Hugging Face Transformers / Datasets
- sentence-transformers or an equivalent embedding-based retrieval library
- Flask (for the web demo)
- Docker (for running GROBID)

#### Install Python dependencies (update the file name as needed):

```bash
pip install -r requirements.txt
```

Dataset  
We crawled and manually annotated a subset of dataset-related papers for training and evaluating BERT-Gate and BERT-Desc, and released it on Hugging Face:

Labeled dataset: https://huggingface.co/datasets/hug2you/Dataset_Collection

### Recommended usage

- Create a local directory, e.g., `./dataset/`.
- Download the dataset from Hugging Face into this directory (you can also load it online via the datasets library).
- In training scripts or config files, point the data path to `./dataset/Dataset_Collection/...`.

### Models

#### Pretrained backbones (not included in this repo, only listed here)

These models can be downloaded automatically from Hugging Face or manually saved under `./models/`:

#### BERT backbone (for initializing BERT-Gate and BERT-Desc)

- `google-bert/bert-base-uncased`

#### Embedding model for retrieval

- `Alibaba-NLP/gte-large-en-v1.5`

#### Long-context baselines

- Longformer: e.g., `allenai/longformer-base-4096`
- BigBird: e.g., `google/bigbird-roberta-base`

In most cases, you only need `bert-base-uncased` and `gte-large-en-v1.5` to run the main system.  
Longformer / BigBird are only required if you want to reproduce the baseline comparisons in the paper.

Your fine-tuned models (highlighted in this README)  
Two core models fine-tuned from `bert-base-uncased` have been released under your Hugging Face account:

**BERT-Gate** (the paper-level gate)

- Repo: https://huggingface.co/hug2you/Bert-Gate  
  Purpose: given the title + abstract, decide whether an arXiv paper introduces a dataset.

**BERT-Desc** (the sentence-level description extractor)

- Repo: https://huggingface.co/hug2you/BERT-Desc  
  Purpose: label sentences in GROBID-parsed full text and extract dataset descriptions.

Recommended local directory layout:

```text
models/
  Bert-Gate/                       # weights downloaded from Hugging Face
  BERT-Desc/
  gte-large-en-v1.5/
```

#### (Optional) Longformer / BigBird / DeepSeek-Qwen as baselines

You only need to set `MODEL_DIR` to the paths above in the corresponding scripts.

#### Optional LLM for link extraction

In the dataset-link extraction module we use a DeepSeek model as an optional LLM verifier:

- `DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16`  
  Repo: https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16

By default you can rely purely on rules + scoring to select links.  
To reproduce LLM-assisted experiments, download the model to `./models/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16/` and enable the LLM mode in the config.

### Code structure

The main directories and scripts corresponding to the paper are:

```text
Data_collection/
  script/
    main_workflow/                     # early end-to-end workflow (kept for reference)
      dataset_crawl.py                 # arXiv crawler
      dataset_description_and_link_extract.py
      loop.py
      pred_label.py
      statistics_latex.py

Train_v2/                              # ⭐ final training and system code
  train/
    run_loop_bert_label.py             # main loop: crawl arXiv + filter + extract + index
    retrieval_framework.py             # retrieval + web backend (GTE encoder)
    qwen_contrast.py                   # LLM comparison for link extraction
    model_Longformer.py                # Longformer baseline
    utils_loop.py                      # loop / scheduling utilities
    utils_train.py                     # common training utilities (metrics, saving, logging)
    utils_bertlabel_latex_extract.py   # description and LaTeX link extraction utilities
    ...
  lyn_test_bert_littledataset.py
  model_Longformer.py
  test_qwen.py
  train4000.py                         # training script for BERT-Desc

statistics_arxiv.py                    # statistics of dataset papers over time (Figure 1)
statistics_arxiv_all.py
statistics_arxiv_lyn.py
statistic_num.py
split_data.py                          # split train/valid/test
test_arxiv.py                          # various small test / debug scripts
```

#### Legacy or other experiments

```text
Train_v1/
  Project1/                            # likely early / unrelated experiments; not used by main pipeline
```

If you want a cleaner repo, you can move `Train_v1/`, `Project1/`, and various `test_*.py` files into a `legacy/` or `experiments/` directory, and add a one-line note in the README saying “early experiments, kept for reference only”.

### Running GROBID (for PDF → XML)

The system calls GROBID to convert arXiv PDFs into structured XML, and then performs sentence-level extraction on top.

```bash
# 1. Start the GROBID Docker service

# pull the image
docker pull lfoppiano/grobid:0.8.0

# run the container (host port 8071 mapped to container port 8070)
docker run -t --rm -p 8071:8070 lfoppiano/grobid:0.8.0
```

Check that the API is alive:

```bash
curl -s http://localhost:8071/api/isalive
```

If it returns `true` or a similar message, GROBID is ready.

### Quick start

```bash
# Prepare data and models
# (1) download the Hugging Face dataset under ./dataset/
# (2) download BERT-Gate, BERT-Desc, and gte-large-en-v1.5 under ./models/

# Start GROBID
# (see commands above)

# Run the automatic pipeline
cd Data_collection/script/Train_v2/train/
python run_loop_bert_label.py
```

The script periodically:

- calls the arXiv API to fetch new papers
- uses BERT-Gate to filter papers that introduce datasets
- applies GROBID + BERT-Desc to extract dataset descriptions
- parses LaTeX sources and selects a primary dataset link
- writes the results into JSONL / an index for the retrieval module

To start the web retrieval demo:

```bash
python web_app.py
```

Then open `http://localhost:5000` in your browser to access the configuration panel, crawl controller, and search interface.

### Citation

If you use this project in your research, please cite our paper (BibTeX omitted here).

### Notes

This repository primarily releases the BERT-Gate and BERT-Desc models together with the corresponding data processing and system code.

All pretrained backbone models (BERT, GTE, Longformer, BigBird, etc.) and the DeepSeek models are from third-party repositories.  
Please follow their respective licenses when using them.
