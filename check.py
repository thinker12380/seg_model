"""环境检查脚本：验证依赖、数据路径、模型能否正常构建"""
import sys
import os

print("=" * 50)
print("环境检查")
print("=" * 50)

# 1. 检查核心依赖
try:
    import torch
    print(f"[OK] torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
except ImportError:
    print("[FAIL] torch 未安装")
    sys.exit(1)

try:
    import cv2
    print("[OK] opencv-python")
except ImportError:
    print("[FAIL] opencv-python 未安装")

try:
    import numpy, pandas, tqdm, sklearn
    print("[OK] numpy, pandas, tqdm, scikit-learn")
except ImportError as e:
    print(f"[FAIL] {e}")

try:
    import albumentations
    print("[OK] albumentations")
except ImportError:
    print("[FAIL] albumentations 未安装（必须安装）")

try:
    import timm
    print("[OK] timm")
except ImportError:
    print("[FAIL] timm 未安装（必须安装）")

try:
    import segmentation_models_pytorch as smp
    print("[OK] segmentation-models-pytorch")
except ImportError:
    print("[WARN] segmentation-models-pytorch 未安装（安装后才能使用 unetplusplus 等SMP模型）")

# 2. 检查数据路径
try:
    from dataset import DATA_ROOT, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, SAMPLE_SUBMISSION
    paths = {
        "TRAIN_IMG_DIR": TRAIN_IMG_DIR,
        "TRAIN_MASK_DIR": TRAIN_MASK_DIR,
        "TEST_IMG_DIR": TEST_IMG_DIR,
        "SAMPLE_SUBMISSION": SAMPLE_SUBMISSION,
    }
    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"{'[OK]' if exists else '[FAIL]'} {name}: {path}")

    # 3. 检查样本数量
    train_imgs = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".jpg")]
    train_masks = [f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith(".png")]
    test_imgs = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".jpg")]
    print(f"[INFO] Train images: {len(train_imgs)}, Train masks: {len(train_masks)}, Test images: {len(test_imgs)}")
except Exception as e:
    print(f"[FAIL] 数据路径检查失败: {e}")

# 4. 快速构建SegNeXtLite验证
try:
    from models import SegNeXtLite
    model = SegNeXtLite(num_classes=4)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    assert out.shape == (1, 4, 512, 512)
    print("[OK] SegNeXtLite 前向传播正常")
except Exception as e:
    print(f"[FAIL] 模型测试失败: {e}")

print("=" * 50)
print("检查完成，如无 [FAIL] 项即可开始训练")
print("=" * 50)
