# train.py  —— 一次性跑 BERT / BigBird / Longformer，并输出三模型对比图
import os
import json
import torch
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import csv
import matplotlib.pyplot as plt

# ======================= 计时/日志辅助 =======================
def _sync_cuda(device):
    try:
        is_cuda = (getattr(device, "type", None) == "cuda") or (isinstance(device, str) and str(device).startswith("cuda"))
        if torch.cuda.is_available() and is_cuda:
            torch.cuda.synchronize()
    except Exception:
        pass

def _now(device):
    _sync_cuda(device)
    return time.perf_counter()

def _append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- 使用原始的 utils 加载逻辑（不要在本文件重写 build_backbone_model） ---
from utils.utils_train import (
    set_seed, get_device, load_json_papers,
    build_tokenizer, build_backbone_model, SentenceClassifier,
    make_dataloader, train_one_epoch, evaluate,
    save_predictions_to_json, plot_history, make_experiment_dir
)

# ======================= 全局数据与通用训练设置 =======================
TRAIN_JSON = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json"
VAL_JSON   = None         # 若你有单独验证集，填路径；否则为 None 表示从训练集切分
VAL_RATIO  = 0.1
SEED       = 42

# ======================= 三个模型的独立配置 =======================
EXPERIMENTS = [
    # 1) BERT：与第一份脚本等价（recall 选优，保存 HF 目录+分类头）
    dict(
        name="bert",
        tag="exp1",
        model_path="/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased",
        out_root="/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs",
        batch_size=1,
        max_length=512,
        min_sent=3,
        context=2,
        epochs=10,
        lr=2e-5,
        weight_decay=0.01,
        grad_accum=4,
        use_amp=False,
        select_metric="recall",
        save_format="hf_head"
    ),

    # 2) BigBird：与第二份脚本等价（F1 选优，保存 state_dict）
    dict(
        name="bigbird",
        tag="bigbird-exp",
        model_path="/home/kzlab/muse/Savvy/Data_collection/models/models--google--bigbird-roberta-base/snapshots/5a145f7852cba9bd431386a58137bf8a29903b90",
        out_root="/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/bigbird",
        batch_size=1,
        max_length=1024,  # 原脚本 1024；显存够可改 4096
        min_sent=3,
        context=2,
        epochs=5,
        lr=2e-5,
        weight_decay=0.01,
        grad_accum=4,
        use_amp=True,
        select_metric="f1",
        save_format="state_dict"
    ),

    # 3) Longformer：与第三份脚本等价（F1 选优，保存 state_dict）
    dict(
        name="longformer",
        tag="longformer-exp",
        model_path="/home/kzlab/muse/Savvy/Data_collection/models/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895",
        out_root="/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/all",
        batch_size=1,
        max_length=4096,
        min_sent=7,
        context=6,
        epochs=5,
        lr=2e-5,
        weight_decay=0.01,
        grad_accum=4,
        use_amp=True,
        select_metric="f1",
        save_format="state_dict"
    ),
]

# ======================= 辅助函数 =======================
def _metric_key(k):
    k = k.lower()
    m = {"acc": "acc", "precision": "prec", "prec": "prec", "recall": "recall", "f1": "f1"}
    return m[k]

def _save_best(exp, backbone, tokenizer, model, save_dir, va_metrics, best_score, epoch, window_hparams):
    key = _metric_key(exp["select_metric"])
    cur = va_metrics[key]
    if cur <= best_score:
        return best_score, False

    best_score = cur

    if exp["save_format"] in ("state_dict", "both"):
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    if exp["save_format"] in ("hf_head", "both"):
        hf_dir = os.path.join(save_dir, "best_hf")
        os.makedirs(hf_dir, exist_ok=True)
        # 1) backbone + config
        backbone.save_pretrained(hf_dir)
        # 2) tokenizer
        tokenizer.save_pretrained(hf_dir)
        # 3) classifier head
        torch.save(model.classifier.state_dict(), os.path.join(hf_dir, "classifier_head.bin"))
        # 4) 元信息
        head_cfg = {
            "num_labels": 2,
            "hidden_size": backbone.config.hidden_size,
            "sep_token_id": tokenizer.sep_token_id,
            "sep_token": tokenizer.sep_token,
            "select_metric": exp["select_metric"],
            "best_score": best_score,
            "epoch": epoch,
            "window_hparams": window_hparams,
            "model_name_or_path": exp["model_path"],
            "train_json": TRAIN_JSON,
            "val_json": VAL_JSON
        }
        with open(os.path.join(hf_dir, "sentence_head_config.json"), "w", encoding="utf-8") as f:
            json.dump(head_cfg, f, ensure_ascii=False, indent=2)

    print(f"[{exp['name']} | epoch {epoch}] 🎯 new best {exp['select_metric'].upper()}={best_score:.4f} -> saved ({exp['save_format']})")
    return best_score, True

# ======================= 单模型训练与计时 =======================
def run_one_experiment(exp, train_papers, val_papers, device):
    # 2) tokenizer/backbone/model
    tokenizer = build_tokenizer(exp["model_path"])
    backbone  = build_backbone_model(exp["model_path"])
    model     = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2).to(device)

    # 3) dataloaders
    train_loader = make_dataloader(
        train_papers, tokenizer,
        batch_size=exp["batch_size"],
        max_length=exp["max_length"],
        min_sentences=exp["min_sent"],
        context_size=exp["context"],
        shuffle=True
    )
    val_loader = make_dataloader(
        val_papers, tokenizer,
        batch_size=exp["batch_size"],
        max_length=exp["max_length"],
        min_sentences=exp["min_sent"],
        context_size=exp["context"],
        shuffle=False
    )

    # 4) optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=exp["weight_decay"])

    # 5) output dir
    dataset_tag = os.path.basename(TRAIN_JSON.rstrip("/"))
    save_dir = make_experiment_dir(exp["out_root"], exp["model_path"], dataset_tag=dataset_tag, tag=exp["tag"])
    print(f"[{exp['name']}] save_dir = {save_dir}")

    # 6) training loop + 速度记录
    history = {k: [] for k in ["train_acc","train_prec","train_recall","train_f1","val_acc","val_prec","val_recall","val_f1"]}
    # 新增的时间/速度指标
    history.update({"train_time_s": [], "eval_time_s": [], "eval_ms_per_item": [], "eval_items_per_s": []})

    best_score = -1.0
    best_epoch = -1
    speed_log_path = os.path.join(save_dir, "speed_log.jsonl")

    for epoch in range(1, exp["epochs"] + 1):
        print(f"\n===== {exp['name']} | Epoch {epoch}/{exp['epochs']} =====")

        # ---- 训练计时 ----
        t0 = _now(device)
        tr = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_accum_steps=exp["grad_accum"], use_amp=exp["use_amp"]
        )
        train_time = _now(device) - t0
        history["train_time_s"].append(train_time)

        # ---- 验证(推理)计时 ----
        t1 = _now(device)
        va, preds = evaluate(model, val_loader, device)
        eval_time = _now(device) - t1
        history["eval_time_s"].append(eval_time)

        # 验证样本规模
        try:
            n_val = len(getattr(val_loader, "dataset", val_papers))
        except Exception:
            n_val = len(val_papers)

        eval_ms_per_item = (eval_time / max(n_val, 1)) * 1000.0
        eval_items_per_s = n_val / max(eval_time, 1e-9)
        history["eval_ms_per_item"].append(eval_ms_per_item)
        history["eval_items_per_s"].append(eval_items_per_s)

        # 指标历史
        for k in ["acc","prec","recall","f1"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])

        # 保存最优
        window_hparams = {"max_length": exp["max_length"], "min_sentences": exp["min_sent"], "context_size": exp["context"]}
        prev_best = best_score
        best_score, saved = _save_best(exp, backbone, tokenizer, model, save_dir, va, best_score, epoch, window_hparams)
        if saved and best_score != prev_best:
            best_epoch = epoch

        # 单模型曲线与预测保存
        plot_history(history, os.path.join(save_dir, "plots"), epoch)
        save_predictions_to_json(preds, os.path.join(save_dir, "predictions", f"preds_epoch_{epoch}.json"))

        # 速度日志（逐行 JSON，便于后处理）
        _append_jsonl(speed_log_path, {
            "model": exp["name"],
            "epoch": epoch,
            "time": datetime.now().isoformat(timespec="seconds"),
            "train_time_s": round(train_time, 4),
            "eval_time_s": round(eval_time, 4),
            "n_val": int(n_val),
            "eval_ms_per_item": round(eval_ms_per_item, 3),
            "eval_items_per_s": round(eval_items_per_s, 3),
        })

        print(f"[{exp['name']} | epoch {epoch}] train={tr}  val={va}  "
              f"[⏱ train {train_time:.2f}s | eval {eval_time:.2f}s | "
              f"{eval_ms_per_item:.2f} ms/item | {eval_items_per_s:.2f} item/s]")

    # 汇总：总推理时长与平均单样本时延（跨所有 epoch）
    n_val_base = len(getattr(val_loader, "dataset", val_papers))
    total_eval_s = sum(history["eval_time_s"])
    total_val_items = n_val_base * len(history["eval_time_s"])
    avg_eval_ms_all_epochs = (total_eval_s / max(total_val_items, 1)) * 1000.0

    # 速度摘要
    speed_summary = {
        "model": exp["name"],
        "best_epoch": int(best_epoch),
        "last_epoch_eval_ms_per_item": float(history["eval_ms_per_item"][-1]),
        "avg_eval_ms_per_item_over_epochs": float(avg_eval_ms_all_epochs),
        "total_eval_time_s_over_epochs": float(total_eval_s),
        "total_train_time_s_over_epochs": float(sum(history["train_time_s"])),
    }

    # 落盘摘要
    with open(os.path.join(save_dir, "speed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(speed_summary, f, ensure_ascii=False, indent=2)

    return dict(
        name=exp["name"],
        history=history,
        save_dir=save_dir,
        select_metric=exp["select_metric"],
        speed_summary=speed_summary
    )

# ======================= 多模型对比图与导出 =======================
def plot_multi_histories(runs, out_png):
    """
    将三模型的 val 曲线画在同一张图上（acc/prec/recall/f1 四个子图）。
    """
    metrics = [("val_acc","Accuracy"), ("val_prec","Precision"), ("val_recall","Recall"), ("val_f1","F1")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        for run in runs:
            y = run["history"][key]
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, label=run["name"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[multi-plot] saved to {out_png}")

def save_speed_csv(runs, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "best_epoch", "last_epoch_eval_ms_per_item",
                         "avg_eval_ms_per_item_over_epochs", "total_eval_time_s_over_epochs", "total_train_time_s_over_epochs"])
        for r in runs:
            s = r["speed_summary"]
            writer.writerow([
                s["model"], s["best_epoch"],
                f'{s["last_epoch_eval_ms_per_item"]:.3f}',
                f'{s["avg_eval_ms_per_item_over_epochs"]:.3f}',
                f'{s["total_eval_time_s_over_epochs"]:.3f}',
                f'{s["total_train_time_s_over_epochs"]:.3f}',
            ])

def plot_speed_comparison(runs, out_png):
    """
    三模型最后一个 epoch 的验证平均单样本时延（ms/item）柱状对比图。
    """
    names = [r["name"] for r in runs]
    ms = [r["speed_summary"]["last_epoch_eval_ms_per_item"] for r in runs]

    plt.figure(figsize=(7,4))
    plt.bar(names, ms)  # 不指定颜色，默认
    plt.ylabel("Validation latency (ms / item)")
    plt.title("Per-model validation latency (last epoch)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[speed-plot] saved to {out_png}")

# ======================= 主流程 =======================
def main():
    # 固定随机种子与设备
    set_seed(SEED)
    device = get_device()

    # 1) 读数据 & 切分/或显式验证集
    print("Loading data ...")
    all_papers = load_json_papers(TRAIN_JSON)
    if VAL_JSON and os.path.isfile(VAL_JSON):
        print("Using explicit validation json.")
        train_papers = all_papers
        val_papers   = load_json_papers(VAL_JSON)
    else:
        print("Splitting data into train/validation ...")
        train_papers, val_papers = train_test_split(all_papers, test_size=VAL_RATIO, random_state=SEED)
    print(f"Train = {len(train_papers)}, Val = {len(val_papers)}")

    # 2) 顺序跑三个模型
    all_runs = []
    for exp in EXPERIMENTS:
        all_runs.append(run_one_experiment(exp, train_papers, val_papers, device))

    # 3) 画一张三模型对比图（四个指标的 val 曲线）
    combined_root = os.path.abspath("/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/all")
    out_png = os.path.join(combined_root, "val_metrics_bert_bigbird_longformer.png")
    plot_multi_histories(all_runs, out_png)

    # 4) 导出速度对比（柱状）与 CSV 摘要
    speed_png = os.path.join(combined_root, "latency_ms_per_item_last_epoch.png")
    plot_speed_comparison(all_runs, speed_png)

    speed_csv = os.path.join(combined_root, "speed_summary.csv")
    save_speed_csv(all_runs, speed_csv)

if __name__ == "__main__":
    main()
