# 手机屏幕表面缺陷分割 — 刷榜训练与推理指南

本项目针对 **MSD 手机屏幕表面缺陷分割** 竞赛（Oil / Stain / Scratch 三类缺陷），提供一套完整的刷榜训练 Pipeline，包含 K-Fold 交叉验证、伪标签半监督、TTA 多尺度推理与多模型集成。

---

## 1. 环境配置

### 1.1 创建/激活 Conda 环境（model）

```bash
conda activate model
```

### 1.2 安装依赖

```bash
pip install albumentations timm segmentation-models-pytorch
```

若首次部署，也可一键安装全部依赖：

```bash
pip install -r requirements.txt
```

### 1.3 环境自检

```bash
python check.py
```

预期输出（带 `[OK]` 即为正常）：

```
[OK] torch 2.7.1+cu128, CUDA=True
[OK] opencv-python
[OK] numpy, pandas, tqdm, scikit-learn
[OK] albumentations
[OK] timm
[OK] segmentation-models-pytorch
[OK] SegNeXtLite 前向传播正常
```

---

## 2. 目录结构

```
d:\seg_model\
├── dataset.py              # DefectDataset + Albumentations 数据增强
├── models.py               # SegNeXtLite / UnetPlusPlus / DeepLabV3+ 等模型定义
├── utils.py                # RLE 编解码、ComboLoss(Dice+BCE+Focal)、mIoU 计算
├── train.py                # 5-Fold 训练 + 伪标签生成
├── train_pseudo.py         # 第二轮：伪标签全量微调
├── predict.py              # TTA 多尺度推理 + 多模型集成 + submission.csv 生成
├── check.py                # 环境与数据路径自检
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
```

---

## 3. 数据说明

| 数据集 | 路径 | 格式 | 说明 |
|--------|------|------|------|
| 训练图像 | `sai-msd2026/dataset_msd/dataset_msd/train/images/` | `.jpg` | 原始手机屏图像 |
| 训练标注 | `sai-msd2026/dataset_msd/dataset_msd/train/masks/` | `.png` | 像素值：0=背景, 38=Oil, 75=Stain, 113=Scratch |
| 测试图像 | `sai-msd2026/dataset_msd/dataset_msd/test/images/` | `.jpg` | 待推理图像 |
| 提交样例 | `sai-msd2026/dataset_msd/dataset_msd/test/sample_submission.csv` | `.csv` | RLE 提交格式样例 |

**类别映射表（代码中已自动处理）：**

| 原始像素值 | 类别索引 | 缺陷类型 |
|-----------|---------|---------|
| 0         | 0       | 背景 (Background) |
| 38        | 1       | Oil |
| 75        | 2       | Stain |
| 113       | 3       | Scratch |

---

## 4. 训练流程

### 4.1 第一轮：K-Fold 训练 + 生成伪标签

```bash
python train.py
```

**执行逻辑：**
1. 将全部 30 张训练样本按 5-Fold 划分。
2. 每 Fold 独立训练一个模型，保存最优 checkpoint 至 `checkpoints/best_fold{i}.pth`。
3. 5 个模型全部训练完毕后，对测试集进行 **概率投票平均**，生成伪标签并保存到 `pseudo_labels/`。

**关键参数（在 `dataset.py` 的 `TRAIN_CONFIG` 中修改）：**

```python
TRAIN_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_folds": 5,
    "img_size": (1024, 1024),   # 可改小为 (512,512) 显存不足时
    "batch_size": 2,             # 根据显存调整
    "epochs": 120,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "save_dir": r"d:\seg_model\checkpoints",
    "pseudo_dir": r"d:\seg_model\pseudo_labels",
    "min_resize": 0.75,
    "max_resize": 1.25,
}
```

**显存优化建议：**
- `batch_size=2` + `img_size=(1024,1024)` 约需 **8-10 GB** 显存。
- 若显存不足，改为 `batch_size=1` 或 `img_size=(512,512)`。

---

### 4.2 第二轮：伪标签全量微调（刷榜核心步骤）

```bash
python train_pseudo.py
```

**执行逻辑：**
1. 加载 `checkpoints/best_fold0.pth` 做 **Warm-start**。
2. 将 30 张真实训练集 + 240 张伪标签合并为一个大数据集。
3. 用 **更低学习率**（`lr * 0.5`）和 **更少 Epoch**（`epochs * 0.6`）做全量 fine-tune。
4. 保存最终模型为 `checkpoints/best_pseudo.pth`。

**为什么能提分？**
- 伪标签相当于把测试集转化为“半监督训练数据”，让模型学到测试域的分布。
- 多模型概率投票生成的伪标签噪声低，比单模型伪标签更可靠。

---

## 5. 推理与提交

### 5.1 基础推理（单模型 + TTA）

```bash
python predict.py
```

**默认行为：**
- 自动读取 `checkpoints/` 下所有 `best_fold*.pth`。
- 若存在多个 checkpoint，自动启用 **模型集成**。
- TTA 包含多尺度 `[0.75, 1.0, 1.25]` + 水平翻转。
- 输出 `submission.csv` 至项目根目录。

### 5.2 仅使用伪标签模型推理

修改 `predict.py` 的 `model_paths` 参数：

```python
model_paths = ["checkpoints_segNext/best_pseudo.pth"]
```

### 5.3 推理参数速查

| 参数 | 说明 |
|------|------|
| `use_tta=True` | 开启多尺度 TTA（默认开启） |
| `use_ensemble=True` | 多模型概率平均（默认开启，仅当 checkpoint >1 时生效） |
| `scales=[0.75, 1.0, 1.25]` | TTA 缩放比例，可加入 `1.5` 或 `0.5` 进一步刷分 |

---

## 6. 核心刷榜策略与调参指南

### 6.1 损失函数权重调优

`utils.py` 中的 `ComboLoss` 默认权重：

```python
ComboLoss(weights=[1.0, 1.0, 2.0])  # [Dice, BCE, Focal]
```

**调参建议：**
- 若某类缺陷 recall 低（漏检多）：**提高 Focal 权重**（如 `3.0`）。
- 若边缘分割粗糙：**提高 Dice 权重**（如 `2.0`）。
- 若类别整体预测概率偏低：**提高 BCE 权重**（如 `2.0`）。

### 6.2 数据增强强度

`dataset.py` 中的 `get_train_transforms` 已包含强增强：

- `RandomResizedCrop`（多尺度训练核心）
- `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`
- `ShiftScaleRotate`（平移+缩放+旋转）
- `ElasticTransform`, `GridDistortion`, `OpticalDistortion`（形变类，对缺陷分割极有效）
- `RandomBrightnessContrast`, `GaussNoise`, `GaussNoise`, `ISONoise`
- `CoarseDropout`（随机遮挡，增强鲁棒性）

**若过拟合严重**：增大 `CoarseDropout` 的 `max_holes` 和 `max_height`。
**若欠拟合**：去掉 `OneOf([GaussNoise, ISONoise], p=0.3)` 中概率，始终启用。

### 6.3 学习率调度策略

当前使用 **CosineAnnealingWarmRestarts**：

```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

- `T_0=10`：每 10 个 Epoch 完成一次余弦退火周期。
- `T_mult=2`：每个周期的长度翻倍（10 -> 20 -> 40...）。

**替代方案（如需更稳定收敛）：**

```python
# 带 warmup 的 CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
```

### 6.4 模型架构选择

| 模型 | 推荐场景 | 修改方式 |
|------|---------|---------|
| **UnetPlusPlus + EfficientNetV2-L** | 稳定基线，预训练强 | `build_model(name="unetplusplus", num_classes=4)` |
| **DeepLabV3+ + EfficientNetV2-L** | 大感受野，适合大面积 Stain | `build_model(name="deeplabv3plus", num_classes=4)` |
| **SegNeXtLite** | 轻量实验，MSCA注意力对细长 Scratch 友好 | `build_model(name="segnext_lite", num_classes=4)` |
| **ViT-Base** | 全局建模强，适合大尺度缺陷关系建模 | `build_model(name="vit_base", num_classes=4)` |
| **ViT-Small** | 参数量更小，便于快速实验 | `build_model(name="vit_small", num_classes=4)` |

**ViT 分割模型说明**

`models.py` 中已实现纯 PyTorch 版 **Vision Transformer (ViT)** 分割模型：
- **Patch Embedding**：`Conv2d(kernel_size=16, stride=16)` 将图像切分为 patch 并投影为向量。
- **Transformer Encoder**：标准 LN + Multi-Head Self-Attention + MLP，堆叠 12 层。
- **Decoder**：ViT 输出的 patch tokens 去掉 `cls_token` 后 reshape 为 2D 特征图，接轻量 CNN Decoder（Conv-BN-ReLU + Upsample）逐步上采样回原图分辨率。
- **特点**：不使用显式可学习位置编码（实验表明影响有限），天然支持任意分辨率输入（需为 `patch_size=16` 的整数倍）。

**注意**：标准 ViT 仅输出单尺度特征，且下采样率固定为 16，对精细边缘分割可能不如 CNN-Based 模型。推荐作为 **多模型集成** 的一员，利用其全局建模能力弥补 CNN 局部感知的不足。

**刷榜终极策略：多模型结果集成**

不同架构对不同类型缺陷的敏感度不同。可分别训练 UnetPlusPlus、DeepLabV3+、SegNeXtLite、ViT-Base，在 `predict.py` 中将它们的概率图做 **加权平均**（权重根据验证集 mIoU 分配）。

---

## 7. 性能监控与 Debug

### 7.1 查看训练日志

每个 Fold 训练时会输出：

```
Fold 0 Epoch 1: Loss=1.2345, Val mIoU=0.4567
  -> Saved best model to checkpoints/best_fold0.pth
```

**关键指标：**
- `Val mIoU`：验证集平均交并比，**越高越好**。
- 若 `Val mIoU` 连续 10 个 Epoch 不提升，可提前终止（加入 EarlyStopping）。

### 7.2 显存不足 (CUDA Out of Memory)

解决方案（按优先级）：
1. 减小 `img_size`：`1024 -> 768 -> 512`
2. 减小 `batch_size`：`4 -> 2 -> 1`
3. 关闭 AMP（混合精度）：在 `train.py` 中注释掉 `torch.cuda.amp.autocast()` 和 `GradScaler`
4. 换用更小的 Encoder：`tu-tf_efficientnetv2_s` 或 `tu-mobilenetv3_large_100`

### 7.3 伪标签质量检查

训练完第一轮后，可人工抽查 `pseudo_labels/` 下的 `.png`：

```bash
# 查看伪标签分布
python -c "
import cv2, numpy as np, os
from collections import Counter
for f in os.listdir('pseudo_labels')[:10]:
    m = cv2.imread(f'pseudo_labels/{f}', cv2.IMREAD_GRAYSCALE)
    print(f, Counter(m.flatten()))
"
```

若某张图全部为 `0`（背景），说明该图置信度低，属于“困难样本”，在第二轮训练时会被自然降权。

---

## 8. 快速命令速查

```bash
# 1. 环境检查
python check.py

# 2. 第一轮 K-Fold 训练 + 伪标签
python train.py

# 3. 第二轮 伪标签微调
python train_pseudo.py

# 4. TTA 推理 + 生成提交文件
python predict.py

# 5. 全流程串行执行（推荐比赛截止前最后一跑）
python train.py && python train_pseudo.py && python predict.py
```

---

## 9. 关键代码入口

| 需求 | 文件 | 关键函数/行 |
|------|------|------------|
| 修改训练参数 | `dataset.py` | `TRAIN_CONFIG` |
| 修改数据增强 | `dataset.py` | `get_train_transforms()` |
| 修改损失权重 | `utils.py` | `ComboLoss.__init__()` |
| 切换模型架构 | `train.py` / `train_pseudo.py` / `predict.py` | `build_model(name="...")` |
| 修改 TTA 尺度 | `predict.py` | `scales=[0.75, 1.0, 1.25]` |
| 调整学习率 | `train.py` / `train_pseudo.py` | `lr=config["lr"]` |

---

## 10. 预期效果与优化空间

**基线预期（UnetPlusPlus + 单模型）：**
- 在 30 张样本上 5-Fold 验证 mIoU 约 **0.55 - 0.65**。
- TTA + 5-Fold 集成后，测试集可再提 **3-5%**。

**刷榜天花板优化点：**
1. **更大分辨率**：若显存允许，`img_size=(1280, 1280)` 或 `(1536, 1536)`。
2. **更多 TTA 尺度**：加入 `0.5, 1.5, 2.0`。
3. **Pseudo Label 迭代**：第二轮训完后，再用新模型重新生成伪标签，迭代 2-3 轮。
4. **多架构 Ensemble**：UnetPlusPlus + DeepLabV3+ + SegNeXt 概率加权。
5. **Test-Time Fine-tuning (TTF)**：对测试单图做几步梯度下降自适应（进阶）。

---

**祝刷榜顺利！**
