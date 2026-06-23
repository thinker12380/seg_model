import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= 配置区域 =================
DATA_ROOT = r"d:\seg_model\sai-msd2026\dataset_msd\dataset_msd"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "test", "sample_submission.csv")

# 原始mask像素值 -> 类别索引
# 0: background, 38: Oil, 75: Stain, 113: Scratch
CLASS_MAPPING = {0: 0, 38: 1, 75: 2, 113: 3}
CLASS_NAMES = ["background", "Oil", "Scratch", "Stain"]
NUM_CLASSES = 4

# 图像尺寸
ORIGIN_H, ORIGIN_W = 1080, 1920

# 训练参数
TRAIN_CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "batch_size": 2,
    "num_workers": 4,
    "epochs": 120,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": r"d:\seg_model\checkpoints",
    "pseudo_dir": r"d:\seg_model\pseudo_labels",
    "img_size": (512, 512),  # (H, W) 训练patch尺寸
    "min_resize": 0.75,
    "max_resize": 1.25,
}

# TTA 尺度
tta_scales = [0.75, 1.0, 1.25]


def get_train_transforms(img_size=(1024, 1024)):
    """强数据增强，针对极少样本刷榜"""
    return A.Compose([
        A.RandomResizedCrop(size=img_size, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size[0]//20, max_width=img_size[1]//20, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=(1024, 1024)):
    return A.Compose([
        A.Resize(*img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class DefectDataset(Dataset):
    def __init__(self, image_ids, img_dir, mask_dir=None, transforms=None, is_test=False, pseudo_dir=None):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.is_test = is_test
        self.pseudo_dir = pseudo_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_test:
            # 测试模式：如有伪标签则加载，否则返回空mask
            if self.pseudo_dir and os.path.exists(os.path.join(self.pseudo_dir, f"{img_id}.png")):
                mask = cv2.imread(os.path.join(self.pseudo_dir, f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            return img, mask, img_id
        else:
            mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 将原始像素值映射到类别索引
            mask_mapped = np.zeros_like(mask, dtype=np.uint8)
            for raw_val, cls_idx in CLASS_MAPPING.items():
                mask_mapped[mask == raw_val] = cls_idx

            if self.transforms:
                augmented = self.transforms(image=img, mask=mask_mapped)
                img = augmented["image"]
                mask = augmented["mask"].long()
            return img, mask, img_id