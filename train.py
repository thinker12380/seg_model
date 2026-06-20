import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import DefectDataset, get_train_transforms, get_val_transforms, TRAIN_CONFIG, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, NUM_CLASSES
from models import build_model
from utils import ComboLoss, iou_score, load_model_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(fold, train_ids, val_ids, config, use_pseudo=False):
    device = config["device"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 数据集
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

    # 模型（推荐使用 unetplusplus 或 unetplusplus_maxvit 刷榜）
    model = build_model(name="segnext_lite", num_classes=NUM_CLASSES)
    model = model.to(device)

    # 损失与优化
    criterion = ComboLoss(weights=[1.0, 1.0, 2.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0.0
    best_path = os.path.join(save_dir, f"best_fold{fold}.pth")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{config['epochs']}")
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

        # 验证
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
        print(f"Fold {fold} Epoch {epoch+1}: Loss={avg_loss:.4f}, Val mIoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "iou": val_iou}, best_path)
            print(f"  -> Saved best model to {best_path}")

    print(f"Fold {fold} best mIoU: {best_iou:.4f}")
    return best_path, best_iou


def generate_pseudo_labels(config):
    """
    K-Fold伪标签：用5个fold的模型对test集进行投票平均，生成伪标签。
    """
    device = config["device"]
    test_ids = [f"test_{i:04d}" for i in range(1, 241)]
    test_ds = DefectDataset(test_ids, TEST_IMG_DIR,
                             transforms=get_val_transforms(config["img_size"]), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    pseudo_dir = config["pseudo_dir"]
    os.makedirs(pseudo_dir, exist_ok=True)

    # 收集所有fold的预测
    all_probs = {img_id: None for img_id in test_ids}
    valid_folds = 0

    for fold in range(config["n_folds"]):
        ckpt_path = os.path.join(config["save_dir"], f"best_fold{fold}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skip fold {fold}")
            continue
        valid_folds += 1
        model = build_model(name="unetplusplus", num_classes=NUM_CLASSES)
        model = load_model_checkpoint(model, ckpt_path).to(device)
        model.eval()

        with torch.no_grad():
            for imgs, _, img_ids in tqdm(test_loader, desc=f"Pseudo fold {fold}"):
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, C, H, W]
                for i, img_id in enumerate(img_ids):
                    if all_probs[img_id] is None:
                        all_probs[img_id] = probs[i]
                    else:
                        all_probs[img_id] += probs[i]

    if valid_folds == 0:
        raise FileNotFoundError("No valid checkpoints found for pseudo labeling!")

    # 平均并保存伪标签
    for img_id in tqdm(test_ids, desc="Saving pseudo labels"):
        if all_probs[img_id] is None:
            continue
        avg_prob = all_probs[img_id] / valid_folds
        pseudo_mask = np.argmax(avg_prob, axis=0).astype(np.uint8)
        # 映射回原始像素值保存
        out_mask = np.zeros_like(pseudo_mask)
        from dataset import CLASS_MAPPING
        reverse_map = {v: k for k, v in CLASS_MAPPING.items()}
        for cls_idx, raw_val in reverse_map.items():
            out_mask[pseudo_mask == cls_idx] = raw_val
        cv2.imwrite(os.path.join(pseudo_dir, f"{img_id}.png"), out_mask)

    print(f"Pseudo labels saved to {pseudo_dir}")


def main():
    set_seed(TRAIN_CONFIG["seed"])
    # 获取所有训练样本ID
    all_ids = sorted([f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".jpg")])
    print(f"Total training samples: {len(all_ids)}")

    # K-Fold训练
    kf = KFold(n_splits=TRAIN_CONFIG["n_folds"], shuffle=True, random_state=TRAIN_CONFIG["seed"])
    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        train_ids = [all_ids[i] for i in train_idx]
        val_ids = [all_ids[i] for i in val_idx]
        print(f"\n========== Fold {fold} ==========")
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
        best_path, best_iou = train_one_fold(fold, train_ids, val_ids, TRAIN_CONFIG)
        results.append((best_path, best_iou))

    print("\n========== K-Fold Results ==========")
    for fold, (p, iou) in enumerate(results):
        print(f"Fold {fold}: {p}, mIoU={iou:.4f}")
    avg_iou = np.mean([iou for _, iou in results])
    print(f"Average mIoU: {avg_iou:.4f}")

    # 生成伪标签（第一轮）
    print("\n========== Generating Pseudo Labels (Round 1) ==========")
    generate_pseudo_labels(TRAIN_CONFIG)

    # 第二轮：加入伪标签重新训练（可选）
    # 如果需要，可以打开下面注释，重新跑main训练
    # print("\n========== Retraining with Pseudo Labels ==========")
    # 将伪标签加入训练集，重新执行train_one_fold


if __name__ == "__main__":
    main()
