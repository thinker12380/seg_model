#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegNeXt_Official 5-Fold 结果差异对比可视化
对比每个fold的预测结果，找出分歧区域
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NUM_CLASSES
from models import build_model
from utils import load_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Prediction Comparison")

    parser.add_argument("--test-dir", type=str, required=True,
                        help="测试图像目录")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="checkpoint目录，包含 best_fold0.pth ~ best_fold4.pth")
    parser.add_argument("--output-dir", type=str, default="fold_compare_output",
                        help="输出目录")
    parser.add_argument("--model", type=str, default="segnext_official",
                        help="模型名称")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="测试图像数量 (0=全部)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[512, 512],
                        metavar=("H", "W"))
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])

    return parser.parse_args()


CLASS_COLORS = {
    0: [0, 0, 0],       # 背景 - 黑
    1: [255, 0, 0],     # Oil - 红
    2: [0, 255, 0],     # Stain - 绿
    3: [0, 0, 255],     # Scratch - 蓝
}

FOLD_COLORS = {
    0: [255, 80, 80],   # 红
    1: [80, 255, 80],   # 绿
    2: [80, 80, 255],   # 蓝
    3: [255, 255, 80],  # 黄
    4: [255, 80, 255],  # 紫
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


def compute_agreement(stacked_preds, num_classes=4):
    """
    计算每个像素有多少个fold预测一致
    stacked_preds: [5, H, W]
    返回: agreement [H, W], 值为 0~1 (1表示全部一致)
    """
    agreement = np.zeros(stacked_preds.shape[1:], dtype=np.float32)
    for cls in range(num_classes):
        agreement += (stacked_preds == cls).sum(axis=0) / stacked_preds.shape[0]
    return agreement


def visualize_fold_comparison(img_name, orig_image, fold_preds, output_dir):
    """
    生成5-Fold对比图
    """
    n_folds = len(fold_preds)
    color_map = get_color_map()

    # 计算共识结果（多数投票）
    stacked = np.stack(list(fold_preds.values()), axis=0)  # [5, H, W]
    consensus = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=NUM_CLASSES).argmax(),
        axis=0, arr=stacked
    )
    agreement = compute_agreement(stacked)

    # ========== 图1: 5个Fold分别预测结果 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原图
    axes[0, 0].imshow(orig_image)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 每个fold的预测
    for idx, (fold_name, pred) in enumerate(fold_preds.items()):
        row = idx // 3
        col = (idx % 3) + 1 if idx < 2 else (idx - 2)
        if idx >= 2:
            row = 1
            col = idx - 2

        pred_color = mask_to_color(pred, color_map)
        axes[row, col].imshow(pred_color)
        fold_num = int(fold_name.replace('best_fold', '').replace('.pth', ''))
        axes[row, col].set_title(f"Fold {fold_num}", fontsize=11,
                                  color=np.array(FOLD_COLORS[fold_num])/255)
        axes[row, col].axis('off')

    # 如果只有5个fold，最后一个位置放共识
    axes[1, 2].imshow(mask_to_color(consensus, color_map))
    axes[1, 2].set_title("Consensus (Majority Vote)", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f"5-Fold Prediction Comparison: {img_name}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_fold_predictions.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    # ========== 图2: 分歧热力图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    # 共识结果
    consensus_color = mask_to_color(consensus, color_map)
    axes[1].imshow(consensus_color)
    axes[1].set_title("Consensus", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 分歧热力图
    disagreement = 1.0 - agreement
    # 只显示有缺陷区域的分歧
    diff_vis = orig_image.copy().astype(np.float32)

    # 用红色深浅表示分歧程度
    disagreement_mask = disagreement > 0
    diff_vis[disagreement_mask, 0] = np.clip(
        diff_vis[disagreement_mask, 0] + disagreement[disagreement_mask] * 255, 0, 255
    )
    diff_vis[disagreement_mask, 1] = diff_vis[disagreement_mask, 1] * 0.5
    diff_vis[disagreement_mask, 2] = diff_vis[disagreement_mask, 2] * 0.5

    im = axes[2].imshow(diff_vis.astype(np.uint8))
    axes[2].set_title("Disagreement Heatmap\n(Red = High Disagreement)", fontsize=12)
    axes[2].axis('off')

    # 添加colorbar
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('disagreement', ['white', 'red'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Disagreement Score', rotation=270, labelpad=15)

    plt.suptitle(f"Fold Agreement Analysis: {img_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_disagreement_heatmap.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    # ========== 图3: 逐类别分歧分析 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    class_names = ["Background", "Oil", "Stain", "Scratch"]

    for cls_idx in range(NUM_CLASSES):
        # 计算每个像素有多少fold预测为这个类别
        class_votes = (stacked == cls_idx).sum(axis=0)  # [H, W]

        # 可视化
        ax = axes[cls_idx]
        ax.imshow(orig_image, alpha=0.3)

        # 用颜色深浅表示投票数
        vote_vis = np.zeros((*class_votes.shape, 4))
        color = np.array(CLASS_COLORS[cls_idx]) / 255.0
        mask = class_votes > 0
        vote_vis[mask, :3] = color
        vote_vis[mask, 3] = class_votes[mask] / n_folds  # alpha = 投票比例

        ax.imshow(vote_vis)
        ax.set_title(f"{class_names[cls_idx]}\n(Votes: 0~{n_folds})", fontsize=11)
        ax.axis('off')

    plt.suptitle(f"Per-Class Fold Voting: {img_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_per_class_voting.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    # 统计信息
    total_pixels = agreement.size
    full_agreement = (agreement == 1.0).sum()
    partial_agreement = ((agreement > 0) & (agreement < 1.0)).sum()

    stats = {
        "full_agreement_ratio": full_agreement / total_pixels,
        "partial_agreement_ratio": partial_agreement / total_pixels,
        "mean_agreement": agreement.mean(),
    }

    return stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"5-Fold Prediction Comparison")
    print(f"模型: {args.model}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"设备: {device}")
    print(f"{'='*60}")

    # 加载5个fold的模型
    fold_models = {}
    for fold in range(5):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_pseudo.pth")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] 未找到: {ckpt_path}")
            continue

        print(f"\n加载 Fold {fold}...")
        model = build_model(name=args.model, num_classes=NUM_CLASSES)
        model = load_model_checkpoint(model, ckpt_path)
        model = model.to(device).eval()
        fold_models[f"best_fold{fold}"] = model
        print(f"  ✓ Fold {fold} 加载成功")

    if len(fold_models) == 0:
        print("错误: 没有加载到任何模型！")
        return

    print(f"\n成功加载 {len(fold_models)} 个模型")

    # 获取测试图片
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(args.test_dir, ext)))

    if args.num_samples > 0 and len(test_images) > args.num_samples:
        np.random.seed(42)
        test_images = np.random.choice(test_images, size=args.num_samples, replace=False).tolist()

    print(f"\n测试图像: {len(test_images)} 张")

    # 收集所有统计
    all_stats = []
    input_size = tuple(args.input_size)

    for img_path in test_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n处理: {img_name}")

        img_tensor, orig_image = preprocess_image(img_path, input_size)

        # 每个fold推理
        fold_preds = {}
        for fold_name, model in fold_models.items():
            pred = inference(model, img_tensor, device)
            fold_preds[fold_name] = pred

        # 可视化对比
        stats = visualize_fold_comparison(img_name, orig_image, fold_preds, args.output_dir)
        stats["image"] = img_name
        all_stats.append(stats)

        print(f"  完全一致像素: {stats['full_agreement_ratio']*100:.1f}%")
        print(f"  平均一致度: {stats['mean_agreement']*100:.1f}%")

    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    avg_full = np.mean([s['full_agreement_ratio'] for s in all_stats])
    avg_mean = np.mean([s['mean_agreement'] for s in all_stats])
    print(f"平均完全一致像素: {avg_full*100:.1f}%")
    print(f"平均一致度: {avg_mean*100:.1f}%")

    # 保存统计到文件
    import json
    with open(os.path.join(args.output_dir, "fold_comparison_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n输出目录: {args.output_dir}")
    print(f"统计文件: {os.path.join(args.output_dir, 'fold_comparison_stats.json')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()