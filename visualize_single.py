#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单个模型 - 可视化预测结果（默认路径版）
直接运行即可，无需命令行参数
"""
import os
import sys
import glob
import argparse
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NUM_CLASSES
from models import build_model
from utils import load_model_checkpoint


# ==================== 默认配置 ====================
# 修改这里即可，无需命令行参数
DEFAULT_CKPT = r"D:\seg_model\fdsnet\checkpoints_fdsnet\fdsnet__phone_voc_best_model.pth"
DEFAULT_MODEL = "fdsnet"
DEFAULT_TEST_DIR = r"D:\seg_model\sai-msd2026\dataset_msd\dataset_msd\test\images"
DEFAULT_OUTPUT_DIR = r"D:\seg_model\test_fdsnet"
DEFAULT_NUM_SAMPLES = 0  # 0=全部
# FDSNet 原始评估使用 1920x1080 * 0.75 -> 1440x810
DEFAULT_INPUT_SIZE = [810, 1440]
DEFAULT_DEVICE = "cuda"
DEFAULT_ALPHA = 0.5
# ==================================================


CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
}


def get_color_map():
    cmap = np.zeros((4, 3), dtype=np.uint8)
    for k, v in CLASS_COLORS.items():
        cmap[k] = v
    return cmap


def mask_to_color(mask, color_map):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        color_mask[mask == cls] = color_map[cls]
    return color_mask


def apply_overlay(image, mask, alpha=0.5):
    overlay = image.copy().astype(np.float32)
    for cls in range(1, NUM_CLASSES):
        cls_mask = (mask == cls)
        if cls_mask.any():
            color = np.array(CLASS_COLORS[cls], dtype=np.float32)
            overlay[cls_mask] = overlay[cls_mask] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((input_size[1], input_size[0]), Image.BILINEAR)
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    orig_image = np.array(image_resized)
    return tensor, orig_image


def inference(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def visualize_single_result(img_name, orig_image, pred_mask, output_dir, alpha=0.5):
    color_map = get_color_map()
    pred_color = mask_to_color(pred_mask, color_map)
    overlay = apply_overlay(orig_image, pred_mask, alpha)

    class_names = ["Background", "Oil", "Stain", "Scratch"]
    pixel_counts = {name: int((pred_mask == i).sum()) for i, name in enumerate(class_names)}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_image)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_color)
    axes[1].set_title("Prediction Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    defect_pixels = [pixel_counts["Oil"], pixel_counts["Stain"], pixel_counts["Scratch"]]
    defect_colors = ['red', 'green', 'blue']
    if sum(defect_pixels) > 0:
        axes[3].pie(defect_pixels, labels=["Oil", "Stain", "Scratch"],
                     colors=defect_colors, autopct='%1.1f%%', startangle=90)
        axes[3].set_title("Defect Distribution", fontsize=12, fontweight='bold')
    else:
        axes[3].text(0.5, 0.5, "No Defect", ha='center', va='center', fontsize=14)
        axes[3].set_title("Defect Distribution", fontsize=12, fontweight='bold')
        axes[3].axis('off')

    plt.suptitle(f"{img_name} | Oil:{pixel_counts['Oil']} Stain:{pixel_counts['Stain']} Scratch:{pixel_counts['Scratch']}",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{img_name}_result.png")
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    Image.fromarray(pred_color).save(os.path.join(output_dir, f"{img_name}_mask.png"))
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{img_name}_overlay.png"))

    return pixel_counts


def main():
    # 使用默认配置
    ckpt = DEFAULT_CKPT
    model_name = DEFAULT_MODEL
    test_dir = DEFAULT_TEST_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    num_samples = DEFAULT_NUM_SAMPLES
    input_size = tuple(DEFAULT_INPUT_SIZE)
    device_str = DEFAULT_DEVICE
    alpha = DEFAULT_ALPHA

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    ckpt_name = os.path.basename(ckpt)

    print(f"{'='*60}")
    print(f"Single Model Test")
    print(f"Checkpoint: {ckpt}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    print(f"加载模型...")
    model = build_model(name=model_name, num_classes=NUM_CLASSES)
    model = load_model_checkpoint(model, ckpt)
    model = model.to(device).eval()
    print(f"  OK: {ckpt_name}")

    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))

    if num_samples > 0 and len(test_images) > num_samples:
        np.random.seed(42)
        test_images = np.random.choice(test_images, size=num_samples, replace=False).tolist()

    print(f"测试图像: {len(test_images)} 张")

    all_stats = []

    for img_path in test_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        img_tensor, orig_image = preprocess_image(img_path, input_size)
        pred_mask = inference(model, img_tensor, device)

        stats = visualize_single_result(img_name, orig_image, pred_mask,
                                         output_dir, alpha)
        stats["image"] = img_name
        all_stats.append(stats)

        print(f"  {img_name}: Oil={stats['Oil']} Stain={stats['Stain']} Scratch={stats['Scratch']}")

    print(f"{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    total_oil = sum(s['Oil'] for s in all_stats)
    total_stain = sum(s['Stain'] for s in all_stats)
    total_scratch = sum(s['Scratch'] for s in all_stats)
    print(f"总 Oil 像素: {total_oil}")
    print(f"总 Stain 像素: {total_stain}")
    print(f"总 Scratch 像素: {total_scratch}")

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"统计保存: {stats_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
