import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    DefectDataset,
    get_val_transforms,
    TRAIN_CONFIG,
    TEST_IMG_DIR,
    SAMPLE_SUBMISSION,
    NUM_CLASSES,
)
from models import build_model
from utils import load_model_checkpoint


# ==================== 配置区域 ====================
# 修改这两项即可，对同一模型输出做行优先/列优先 RLE A/B 对比。
MODEL_NAME = "unetplusplus"
CHECKPOINT_PATH = r"d:\seg_model\checkpoints_Unet++\best_fold3.pth"

OUTPUT_ROW_MAJOR = r"d:\seg_model\submission_rle_row_major.csv"
OUTPUT_COL_MAJOR = r"d:\seg_model\submission_rle_col_major.csv"
# ==================================================


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU setup.")
    return torch.device("cuda")


def rle_encode(mask, order="F"):
    pixels = mask.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return "0 0"
    return " ".join(str(x) for x in runs)


def predict_prob(model, img, device):
    model.eval()
    original_size = (img.shape[1], img.shape[2])
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(img)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        if probs.shape[2:] != original_size:
            probs = F.interpolate(
                probs,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
    return probs.squeeze(0).cpu().numpy()


def build_submission_rows(model, device, config):
    test_ids = [f"test_{i:04d}" for i in range(1, 241)]
    test_ds = DefectDataset(
        test_ids,
        TEST_IMG_DIR,
        transforms=get_val_transforms(config["img_size"]),
        is_test=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    row_major_rows = []
    col_major_rows = []

    for imgs, _, img_ids in tqdm(test_loader, desc="Predicting"):
        img = imgs[0]
        img_id = img_ids[0]
        prob = predict_prob(model, img, device)
        pred = np.argmax(prob, axis=0).astype(np.uint8)

        for cls_idx in [1, 2, 3]:
            binary_mask = (pred == cls_idx).astype(np.uint8)
            row_major_rows.append(
                {
                    "id": f"{img_id}_{cls_idx}",
                    "rle": rle_encode(binary_mask, order="C"),
                }
            )
            col_major_rows.append(
                {
                    "id": f"{img_id}_{cls_idx}",
                    "rle": rle_encode(binary_mask, order="F"),
                }
            )

    return row_major_rows, col_major_rows


def save_submission(rows, output_path):
    df = pd.DataFrame(rows)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    df = df.set_index("id").reindex(sample["id"]).reset_index()
    df["rle"] = df["rle"].fillna("0 0")
    df.to_csv(output_path, index=False)
    return df


def main():
    print("=" * 60)
    print("RLE A/B Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = get_device()
    config = TRAIN_CONFIG.copy()
    model = build_model(name=MODEL_NAME, num_classes=NUM_CLASSES)
    model = load_model_checkpoint(model, CHECKPOINT_PATH).to(device).eval()

    row_major_rows, col_major_rows = build_submission_rows(model, device, config)

    row_df = save_submission(row_major_rows, OUTPUT_ROW_MAJOR)
    col_df = save_submission(col_major_rows, OUTPUT_COL_MAJOR)

    print(f"Saved row-major submission to: {OUTPUT_ROW_MAJOR}")
    print(f"Saved col-major submission to: {OUTPUT_COL_MAJOR}")
    print(f"Row-major preview:\n{row_df.head(6).to_string(index=False)}")
    print(f"Col-major preview:\n{col_df.head(6).to_string(index=False)}")


if __name__ == "__main__":
    main()
