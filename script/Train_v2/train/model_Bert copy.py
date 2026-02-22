# train.py
import os
import torch
from utils.utils_train import (
    set_seed, get_device, load_json_papers,
    build_tokenizer, build_backbone_model, SentenceClassifier,
    make_dataloader, train_one_epoch, evaluate,
    save_predictions_to_json, plot_history, make_experiment_dir
)
from sklearn.model_selection import train_test_split
import json



def main():
    # ---------------------------- 配置区 ----------------------------
    MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"          # ✅ 例如 "bert-base-cased" 或本地路径
    TRAIN_JSON = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json"         # ✅ 训练集 JSON 文件路径
    TEST_JSON   = None        # ✅ 验证集 JSON 文件路径
    VAL_JSON   = None        # 用 None 表示只有一个文件
    VAL_RATIO  = 0.1         # 验证集占比
    SEED       = 42
    OUT_ROOT   = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs"                    # ✅ 输出保存目录
    TAG        = "exp1"                         # ✅ 可选实验标识

    BATCH_SIZE = 1
    MAX_LENGTH = 512                            # 若为 None，自动取 tokenizer.model_max_length
    MIN_SENT   = 3
    CONTEXT    = 2
    EPOCHS     = 10
    LR         = 2e-5
    WEIGHT_DECAY = 0.01
    GRAD_ACCUM   = 4
    USE_AMP      = False
    # -------------------------------------------------------------

    set_seed(SEED)
    device = get_device()

    # 1) 数据加载
    print("Loading data ...")
    all_papers = load_json_papers(TRAIN_JSON)
    print("Splitting data into train and validation sets...")
    train_papers, val_papers = train_test_split(all_papers, test_size=VAL_RATIO, random_state=SEED)
    print(f"Train set size: {len(train_papers)}, Validation set size: {len(val_papers)}")
    
    # 2) 模型与 tokenizer 初始化
    tokenizer = build_tokenizer(MODEL_PATH)
    backbone  = build_backbone_model(MODEL_PATH)
    model     = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2).to(device)

    # 3) 构建 DataLoader
    train_loader = make_dataloader(train_papers, tokenizer, batch_size=BATCH_SIZE,
                                   max_length=MAX_LENGTH, min_sentences=MIN_SENT,
                                   context_size=CONTEXT, shuffle=True)
    val_loader   = make_dataloader(val_papers, tokenizer, batch_size=BATCH_SIZE,
                                   max_length=MAX_LENGTH, min_sentences=MIN_SENT,
                                   context_size=CONTEXT, shuffle=False)

    # 4) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 5) 输出目录
    save_dir = make_experiment_dir(OUT_ROOT, MODEL_PATH, dataset_tag=os.path.basename(TRAIN_JSON), tag=TAG)

    # 6) 训练循环
    best_recall = 0.0
    history = {k: [] for k in ["train_acc","train_prec","train_recall","train_f1","val_acc","val_prec","val_recall","val_f1"]}

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        # 1 训练
        tr = train_one_epoch(model, train_loader, optimizer, device, grad_accum_steps=GRAD_ACCUM, use_amp=USE_AMP)
        # 2 评估
        va, preds = evaluate(model, val_loader, device)

        for k in ["acc","prec","recall","f1"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])

        if va["recall"] > best_recall: # 把这里改成了recall
            best_recall = va["recall"]
            
            # torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            # 以 Hugging Face 目录形式保存（backbone+tokenizer），分类头单独存
            hf_dir = os.path.join(save_dir, "best_hf")
            os.makedirs(hf_dir, exist_ok=True)

            # 1) 保存 backbone（会写 config.json + pytorch_model.bin 等）
            backbone.save_pretrained(hf_dir)
            # 2) 保存 tokenizer（会写 tokenizer.json/vocab 等）
            tokenizer.save_pretrained(hf_dir)
            # 3) 保存句级分类头（单独一个 bin）
            torch.save(model.classifier.state_dict(), os.path.join(hf_dir, "classifier_head.bin"))
            # 4) 写一份分类头元信息，便于复现与加载
            head_cfg = {
                "num_labels": 2,
                "hidden_size": backbone.config.hidden_size,
                "sep_token_id": tokenizer.sep_token_id,
                "sep_token": tokenizer.sep_token,
                "select_metric": "recall",
                "best_recall": best_recall,
                "epoch": epoch + 1,
                "window_hparams": {
                    "max_length": MAX_LENGTH,
                    "min_sentences": MIN_SENT,
                    "context_size": CONTEXT
                },
                "model_name_or_path": MODEL_PATH,
                "train_json": TRAIN_JSON
            }
            with open(os.path.join(hf_dir, "sentence_head_config.json"), "w", encoding="utf-8") as f:
                json.dump(head_cfg, f, ensure_ascii=False, indent=2)
            
            
            
            
            print(f"[epoch {epoch+1}] 🎯 new best Recall={best_recall:.4f} -> saved")
            
            
        # 3 绘图
        plot_history(history, os.path.join(save_dir, "plots"), epoch+1)
        save_predictions_to_json(preds, os.path.join(save_dir, "predictions", f"preds_epoch_{epoch+1}.json"))
        print(f"[epoch {epoch+1}] train={tr}  val={va}")

if __name__ == "__main__":
    main()
