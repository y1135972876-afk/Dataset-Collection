# train.py
import os
import torch
from utils.utils_train import (
    set_seed, get_device, load_json_papers,
    build_tokenizer, build_backbone_model, SentenceClassifier,
    make_dataloader, evaluate,
    save_predictions_to_json, make_experiment_dir
)



def main():
    # ---------------------------- 配置区 ----------------------------
    MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/bert-base-cased__train_val_revise.json__20251025_230335__exp1/best_hf"          # ✅ 例如 "bert-base-cased" 或本地路径
    # 如果要从你前面保存的 Hugging Face 目录恢复（继续训练或只评估），把它设成 best_hf 目录；否则设为 None
    RESUME_HF_DIR = None  # 例如："/home/xxx/outputs/.../best_hf"
    
    VAL_JSON = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json"         # ✅ 训练集 JSON 文件路径
    SEED       = 42
    OUT_ROOT   = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs"                    # ✅ 输出保存目录
    TAG        = "exp1"                         # ✅ 可选实验标识

    BATCH_SIZE = 1
    MAX_LENGTH = 512                            # 若为 None，自动取 tokenizer.model_max_length
    MIN_SENT   = 3
    CONTEXT    = 2
    # -------------------------------------------------------------

    set_seed(SEED)
    device = get_device()

    # 1) 数据加载
    print("Loading data ...")
    val_papers = load_json_papers(VAL_JSON)  # 也可以改成用 TEST_JSON
    print(f"Validation set size: {len(val_papers)}")
    
    # 2) 加载四要素：tokenzier，backbone（核心bert），head，真题模型
    if RESUME_HF_DIR:
        tokenizer = build_tokenizer(RESUME_HF_DIR)       # 等价于 AutoTokenizer.from_pretrained(RESUME_HF_DIR)
        backbone  = build_backbone_model(RESUME_HF_DIR)  # 等价于 AutoModel.from_pretrained(RESUME_HF_DIR)
    else:
        tokenizer = build_tokenizer(MODEL_PATH)
        backbone  = build_backbone_model(MODEL_PATH)
    #2.3 加载model
    model = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2)
    
    #2.4 加载head
    head_path = os.path.join(MODEL_PATH, "classifier_head.bin")
    head_state = torch.load(head_path, map_location="cpu")
    # 严格加载，防止头没加载上导致指标骤降
    missing, unexpected = model.classifier.load_state_dict(head_state, strict=True)
    assert not missing and not unexpected, f"head missing={missing}, unexpected={unexpected}"

    # 3) 再把整个模型搬到设备
    model = model.to(device)


    # 4) 构建 DataLoader
    val_loader   = make_dataloader(val_papers, tokenizer, batch_size=BATCH_SIZE,
                                   max_length=MAX_LENGTH, min_sentences=MIN_SENT,
                                   context_size=CONTEXT, shuffle=False)

    # ===== 仅评估 =====
    # 5) 输出目录
    save_dir = make_experiment_dir(
        OUT_ROOT,
        RESUME_HF_DIR if RESUME_HF_DIR else MODEL_PATH,
        dataset_tag=os.path.basename(VAL_JSON),  # 若想用 TEST_JSON 做评估，这里也一起改
        tag=TAG
    )

    # ===== 仅评估 =====
    va, preds = evaluate(model, val_loader, device)
    print(f"Eval metrics: {va}")

    # 保存预测结果
    # 如果 utils 里没自动建目录，可先：
    # os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
    save_predictions_to_json(preds, os.path.join(save_dir, "predictions", "preds_eval.json"))





if __name__ == "__main__":
    main()
