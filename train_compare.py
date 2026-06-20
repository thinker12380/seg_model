"""
多模型自动对比训练
用法: python train_compare.py

自动训练多个模型，每个模型跑少量 epoch 快速对比，
记录验证 mIoU，最后输出对比表格。
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from dataset import DefectDataset, get_train_transforms, get_val_transforms, TRAIN_CONFIG
from models import build_model
from utils import ComboLoss, iou_score
from torch.utils.data import DataLoader

TRAIN_IMG_DIR = r"D:\seg_model\sai-msd2026\dataset_msd\dataset_msd\train\images"
TRAIN_MASK_DIR = r"D:\seg_model\sai-msd2026\dataset_msd\dataset_msd\train\masks"
NUM_CLASSES = 4

# 重点对比 Top 3 刷榜模型（准确率优先）
MODELS_TO_COMPARE = [
    ("segnext_official", "官方SegNeXt"),
    ("segmenter_mask", "Segmenter-Mask"),
    ("unetplusplus", "SMP-Unet++"),
]

# 快速对比配置：每个模型只跑少量 epoch
QUICK_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": (512, 512),
    "batch_size": 1,
    "epochs": 20,        # 快速对比：20 epoch
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "save_dir": "checkpoints_compare",
}


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_model(name, desc, config, train_ids, val_ids):
    device = config["device"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"开始训练: {name} ({desc})")
    print(f"{'='*60}")

    train_ds = DefectDataset(
        train_ids, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        transforms=get_train_transforms(config["img_size"]),
    )
    val_ds = DefectDataset(
        val_ids, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        transforms=get_val_transforms(config["img_size"]),
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    model = build_model(name=name, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = ComboLoss(weights=[1.0, 1.0, 2.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0.0
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{config['epochs']}")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                for p, t in zip(preds, targets):
                    val_ious.append(iou_score(p, t, num_classes=NUM_CLASSES, ignore_index=0))

        val_iou = np.mean(val_ious)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_miou": val_iou})
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val mIoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = os.path.join(save_dir, f"best_{name}.pth")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "iou": val_iou}, ckpt_path)
            print(f"  -> Saved best model to {ckpt_path}")

    print(f"{name} 最佳 Val mIoU: {best_iou:.4f}")
    return {
        "name": name,
        "desc": desc,
        "best_miou": best_iou,
        "history": history,
    }


def main():
    set_seed(QUICK_CONFIG["seed"])
    device = QUICK_CONFIG["device"]
    print(f"Device: {device}")

    train_ids = sorted([f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".jpg")])
    print(f"总训练样本: {len(train_ids)}")

    # 简单划分：80% 训练，20% 验证（用于快速对比）
    split = int(len(train_ids) * 0.8)
    train_ids_split = train_ids[:split]
    val_ids_split = train_ids[split:]
    print(f"训练: {len(train_ids_split)}, 验证: {len(val_ids_split)}")

    results = []
    for name, desc in MODELS_TO_COMPARE:
        try:
            result = train_one_model(name, desc, QUICK_CONFIG, train_ids_split, val_ids_split)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "desc": desc, "best_miou": 0.0, "history": [], "error": str(e)})

    # 保存结果
    with open("compare_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印对比表格
    print("\n" + "="*60)
    print("对比结果汇总")
    print("="*60)
    print(f"{'模型':<25} {'描述':<20} {'最佳Val mIoU':<15}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x["best_miou"], reverse=True):
        print(f"{r['name']:<25} {r['desc']:<20} {r['best_miou']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
