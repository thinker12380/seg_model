"""
第二轮训练：将伪标签加入训练集，进行全量训练以提升泛化性能。
同时增加训练epochs和减小学习率，做fine-tune。
"""
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset import DefectDataset, get_train_transforms, TRAIN_CONFIG, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, NUM_CLASSES
from models import build_model
from utils import ComboLoss, iou_score, load_model_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(TRAIN_CONFIG["seed"] + 1)
    device = TRAIN_CONFIG["device"]
    save_dir = TRAIN_CONFIG["save_dir"]
    pseudo_dir = TRAIN_CONFIG["pseudo_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 原始训练集
    train_ids = sorted([f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".jpg")])
    # 伪标签样本（仅选取置信度较高的，这里简单使用全部）
    pseudo_ids = sorted([f.split(".")[0] for f in os.listdir(pseudo_dir) if f.endswith(".png")])
    print(f"Original train: {len(train_ids)}, Pseudo: {len(pseudo_ids)}")

    ds_real = DefectDataset(train_ids, TRAIN_IMG_DIR, TRAIN_MASK_DIR, transforms=get_train_transforms(TRAIN_CONFIG["img_size"]))
    ds_pseudo = DefectDataset(pseudo_ids, TEST_IMG_DIR,
                               mask_dir=pseudo_dir, transforms=get_train_transforms(TRAIN_CONFIG["img_size"]))
    ds_all = ConcatDataset([ds_real, ds_pseudo])
    loader = DataLoader(ds_all, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                        num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True)

    model = build_model(name="unetplusplus", num_classes=NUM_CLASSES).to(device)
    # 可加载上一轮最佳权重做热启动
    init_ckpt = os.path.join(save_dir, "best_fold0.pth")
    if os.path.exists(init_ckpt):
        model = load_model_checkpoint(model, init_ckpt)
        print(f"Warm-start from {init_ckpt}")

    criterion = ComboLoss(weights=[1.0, 1.0, 2.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"] * 0.5, weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    best_path = os.path.join(save_dir, "best_pseudo.pth")

    epochs = int(TRAIN_CONFIG["epochs"] * 0.6)  # 第二轮不用跑满
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks, _ in tqdm(loader, desc=f"Pseudo Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Pseudo Epoch {epoch+1}: Loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss}, best_path)
            print(f"  -> Saved to {best_path}")

    print("Pseudo training done.")


if __name__ == "__main__":
    main()
