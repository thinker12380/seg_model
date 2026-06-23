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


# ==================== 模型配置 ====================
# 单模型推理时使用的模型名称
MODEL_NAME = "unetplusplus"  # 可选: "segnext_official", "unetplusplus" 等

# 多模型集成配置（异架构集成）
# 每个模型指定: checkpoint路径、模型名称、权重
ENSEMBLE_CONFIGS = [

    {
        "path": "checkpoints_Unet++/best_fold3.pth",
        "name": "unetplusplus",
        "weight": 1.0,
    },
]
# 是否启用异架构集成（False则使用单模型）
USE_MULTI_MODEL_ENSEMBLE = False
# ==================================================


def get_cuda_device():
    """获取 CUDA 设备，如果不可用则报错"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU setup.")

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    print(f"[CUDA] Using GPU: {gpu_name}")
    print(f"[CUDA] GPU Memory: {gpu_memory:.2f} GB")
    print(f"[CUDA] CUDA Version: {torch.version.cuda}")
    return device


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

            # 原图 - 使用 CUDA 自动混合精度加速
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


def ensemble_same_model(model_paths, img, device, scales=[0.75, 1.0, 1.25]):
    """同架构多模型集成：对多个fold的模型预测取平均"""
    all_probs = []
    for path in model_paths:
        model = build_model(name=MODEL_NAME, num_classes=NUM_CLASSES)
        model = load_model_checkpoint(model, path).to(device)
        prob = tta_predict(model, img, device, scales=scales)
        all_probs.append(prob)
        # 释放 GPU 内存
        del model
        torch.cuda.empty_cache()
    avg_prob = np.stack(all_probs, axis=0).mean(axis=0)
    return avg_prob


def ensemble_multi_model(img, device, scales=[0.75, 1.0, 1.25]):
    """
    异架构多模型集成：segnext_official + unetplusplus 等
    按权重加权平均概率图
    """
    all_probs = []
    total_weight = 0

    for cfg in ENSEMBLE_CONFIGS:
        if not os.path.exists(cfg["path"]):
            print(f"[WARNING] Checkpoint not found: {cfg['path']}, skipping {cfg['name']}")
            continue

        print(f"[Ensemble] Loading {cfg['name']} from {cfg['path']} (weight={cfg['weight']})")
        model = build_model(name=cfg["name"], num_classes=NUM_CLASSES)

        # 验证 checkpoint 中的模型名称
        checkpoint = torch.load(cfg["path"], map_location="cpu", weights_only=False)
        saved_name = checkpoint.get("model_name", "unknown")
        if saved_name != cfg["name"]:
            print(f"[WARNING] Checkpoint model_name '{saved_name}' != config name '{cfg['name']}'")

        model = load_model_checkpoint(model, cfg["path"])
        model = model.to(device).eval()

        prob = tta_predict(model, img, device, scales=scales)
        all_probs.append(prob * cfg["weight"])
        total_weight += cfg["weight"]

        # 释放 GPU 内存
        del model, checkpoint
        torch.cuda.empty_cache()

    if len(all_probs) == 0:
        raise FileNotFoundError("No valid checkpoints found for ensemble!")

    # 归一化权重
    result = np.stack(all_probs, axis=0).sum(axis=0) / total_weight
    return result


def predict_and_submit(config, model_paths=None, use_tta=True, use_ensemble=True):
    # 强制使用 CUDA
    device = get_cuda_device()

    # 覆盖 config 中的 device 设置
    config["device"] = device

    save_dir = config["save_dir"]

    # 自动获取所有fold checkpoint（单模型集成模式）
    if model_paths is None and not USE_MULTI_MODEL_ENSEMBLE:
        model_paths = [os.path.join(save_dir, f"best_fold{i}.pth") for i in range(config["n_folds"])]
        model_paths = [p for p in model_paths if os.path.exists(p)]

    if not USE_MULTI_MODEL_ENSEMBLE and len(model_paths) == 0:
        raise FileNotFoundError(f"No model checkpoints found in {save_dir}")

    if USE_MULTI_MODEL_ENSEMBLE:
        print(f"Using multi-model ensemble: {[c['name'] for c in ENSEMBLE_CONFIGS]}")
    else:
        print(f"Using checkpoints: {model_paths}")

    test_ids = [f"test_{i:04d}" for i in range(1, 241)]
    test_ds = DefectDataset(
        test_ids,
        TEST_IMG_DIR,
        transforms=get_val_transforms(config["img_size"]),
        is_test=True,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    submissions = []
    reverse_map = {v: k for k, v in CLASS_MAPPING.items()}  # 1->38, 2->75, 3->113

    for imgs, _, img_ids in tqdm(test_loader, desc="Predicting"):
        img = imgs[0]  # [C, H, W]
        img_id = img_ids[0]

        if USE_MULTI_MODEL_ENSEMBLE:
            # 异架构集成
            prob = ensemble_multi_model(img, device, scales=tta_scales if use_tta else [1.0])
        elif use_ensemble and len(model_paths) > 1:
            # 同架构多fold集成
            prob = ensemble_same_model(model_paths, img, device, scales=tta_scales if use_tta else [1.0])
        else:
            # 单模型推理
            model = build_model(name=MODEL_NAME, num_classes=NUM_CLASSES)
            model = load_model_checkpoint(model, model_paths[0]).to(device)
            prob = tta_predict(model, img, device, scales=tta_scales if use_tta else [1.0])
            del model
            torch.cuda.empty_cache()

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
    out_path = os.path.join(os.path.dirname(save_dir), "D:/seg_model/sai-msd2026/submission.csv")
    df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    return df


def main():
    print(f"{'='*60}")
    print(f"Predict - 模型: {MODEL_NAME}")
    print(f"多模型集成: {USE_MULTI_MODEL_ENSEMBLE}")
    print(f"{'='*60}")

    config = TRAIN_CONFIG
    predict_and_submit(config, use_tta=True, use_ensemble=True)


if __name__ == "__main__":
    main()