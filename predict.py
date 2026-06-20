import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import DefectDataset, get_val_transforms, TRAIN_CONFIG, TEST_IMG_DIR, SAMPLE_SUBMISSION, NUM_CLASSES, CLASS_MAPPING, tta_scales
from models import build_model
from utils import rle_encode, load_model_checkpoint


def tta_predict(model, img, device, scales=[0.75, 1.0, 1.25]):
    """
    TTA推理：多尺度 + 水平翻转
    img: [C, H, W] tensor
    """
    model.eval()
    all_probs = []
    original_size = (img.shape[1], img.shape[2])  # (H, W)

    with torch.no_grad():
        for scale in scales:
            if scale != 1.0:
                h = int(img.shape[1] * scale)
                w = int(img.shape[2] * scale)
                resized = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
            else:
                resized = img.unsqueeze(0)
            resized = resized.to(device)

            # 原图
            with torch.cuda.amp.autocast():
                logits = model(resized)
            probs = torch.softmax(logits, dim=1)
            # 插回原尺寸
            if probs.shape[2:] != original_size:
                probs = F.interpolate(probs, size=original_size, mode="bilinear", align_corners=False)
            all_probs.append(probs.cpu())

            # 水平翻转
            flipped = torch.flip(resized, dims=[3])
            with torch.cuda.amp.autocast():
                logits = model(flipped)
            probs = torch.softmax(logits, dim=1)
            probs = torch.flip(probs, dims=[3])
            if probs.shape[2:] != original_size:
                probs = F.interpolate(probs, size=original_size, mode="bilinear", align_corners=False)
            all_probs.append(probs.cpu())

    # 平均所有TTA结果
    avg_prob = torch.stack(all_probs, dim=0).mean(dim=0).squeeze(0)  # [C, H, W]
    return avg_prob.numpy()


def ensemble_predict(model_paths, img, device, scales=[0.75, 1.0, 1.25]):
    """多模型集成：对多个fold的模型预测取平均"""
    all_probs = []
    for path in model_paths:
        model = build_model(name="unetplusplus", num_classes=NUM_CLASSES)
        model = load_model_checkpoint(model, path).to(device)
        prob = tta_predict(model, img, device, scales=scales)
        all_probs.append(prob)
    avg_prob = np.stack(all_probs, axis=0).mean(axis=0)
    return avg_prob


def predict_and_submit(config, model_paths=None, use_tta=True, use_ensemble=True):
    device = config["device"]
    save_dir = config["save_dir"]
    # 自动获取所有fold checkpoint
    if model_paths is None:
        model_paths = [os.path.join(save_dir, f"best_fold{i}.pth") for i in range(config["n_folds"])]
        model_paths = [p for p in model_paths if os.path.exists(p)]

    if len(model_paths) == 0:
        raise FileNotFoundError(f"No model checkpoints found in {save_dir}")
    print(f"Using checkpoints: {model_paths}")

    test_ids = [f"test_{i:04d}" for i in range(1, 241)]
    test_ds = DefectDataset(
        test_ids,
        TEST_IMG_DIR,
        transforms=get_val_transforms(config["img_size"]),
        is_test=True,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    submissions = []
    reverse_map = {v: k for k, v in CLASS_MAPPING.items()}  # 1->38, 2->75, 3->113
    # 注意：提交时_id后缀对应Oil(1), Stain(2), Scratch(3)
    # 我们按 class_idx 1,2,3 顺序对应

    for imgs, _, img_ids in tqdm(test_loader, desc="Predicting"):
        img = imgs[0]  # [C, H, W]
        img_id = img_ids[0]
        if use_ensemble and len(model_paths) > 1:
            prob = ensemble_predict(model_paths, img, device, scales=tta_scales if use_tta else [1.0])
        else:
            model = build_model(name="unetplusplus", num_classes=NUM_CLASSES)
            model = load_model_checkpoint(model, model_paths[0]).to(device)
            prob = tta_predict(model, img, device, scales=tta_scales if use_tta else [1.0])

        # prob shape: [C, H, W]
        pred = np.argmax(prob, axis=0)  # [H, W]
        # 对每类分别编码RLE
        for cls_idx in [1, 2, 3]:  # Oil, Stain, Scratch
            binary_mask = (pred == cls_idx).astype(np.uint8)
            rle = rle_encode(binary_mask)
            submissions.append({
                "id": f"{img_id}_{cls_idx}",
                "rle": rle,
            })

    df = pd.DataFrame(submissions)
    # 按照sample_submission的ID顺序排序
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    df = df.set_index("id").reindex(sample["id"]).reset_index()
    out_path = os.path.join(os.path.dirname(save_dir), "submission.csv")
    df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    return df


def main():
    config = TRAIN_CONFIG
    predict_and_submit(config, use_tta=True, use_ensemble=True)


if __name__ == "__main__":
    main()
