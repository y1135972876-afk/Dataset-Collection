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

def main():
    # ---------------------------- 配置区（改这里） ----------------------------
    # ✅ Longformer 本地快照目录或 HF 名（如 "allenai/longformer-base-4096"）
    MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/models/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895"

    TRAIN_JSON = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/train_val_revise.json"  
    VAL_RATIO  = 0.1
    SEED       = 42
    OUT_ROOT   = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/longformer"
    TAG        = "longformer-exp"

    # ✅ Longformer 支持 4096，上限建议明确为 4096
    BATCH_SIZE = 1
    MAX_LENGTH = 4096
    # ✅ 你给的 longformer 句窗参数
    MIN_SENT   = 7
    CONTEXT    = 6

    EPOCHS     = 5
    LR         = 2e-5
    WEIGHT_DECAY = 0.01
    GRAD_ACCUM   = 4
    USE_AMP      = True   # 若要与旧脚本数值完全对齐，可改 False
    # ------------------------------------------------------------------------

    set_seed(SEED)
    device = get_device()

    # 1) 数据加载 & 切分
    print("Loading data ...")
    all_papers = load_json_papers(TRAIN_JSON)
    print("Splitting data into train and validation sets...")
    train_papers, val_papers = train_test_split(all_papers, test_size=VAL_RATIO, random_state=SEED)
    print(f"Train set size: {len(train_papers)}, Validation set size: {len(val_papers)}")

    # 2) 模型与 tokenizer 初始化（Auto 系列自动拿 Longformer）
    tokenizer = build_tokenizer(MODEL_PATH)
    backbone  = build_backbone_model(MODEL_PATH)  # -> AutoModel(LongformerModel)
    model     = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2).to(device)

    # 3) DataLoader（窗口内用 sep_token 拼接，句级分类头）
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
    best_f1 = 0.0
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

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"[epoch {epoch+1}] 🎯 new best F1={best_f1:.4f} -> saved")

        plot_history(history, os.path.join(save_dir, "plots"), epoch+1)
        save_predictions_to_json(preds, os.path.join(save_dir, "predictions", f"preds_epoch_{epoch+1}.json"))
        print(f"[epoch {epoch+1}] train={tr}  val={va}")

if __name__ == "__main__":
    # （可选）限定使用的物理 GPU：比如只用卡1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
