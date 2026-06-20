import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= RLE 工具 =================
def rle_encode(mask):
    """将单通道二值mask编码为RLE字符串"""
    pixels = mask.flatten(order='F')  # Fortran风格按列展开，符合常见竞赛格式
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return "0 0"
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_str, shape=(1080, 1920)):
    """RLE解码为mask"""
    s = rle_str.strip().split()
    if s == ["0", "0"]:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# ================= 损失函数 =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: [B, C, H, W], targets: [B, H, W]
        num_classes = preds.shape[1]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        preds = torch.sigmoid(preds) if preds.shape[1] == 1 else torch.softmax(preds, dim=1)
        intersection = (preds * targets_onehot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        ce = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()


class ComboLoss(nn.Module):
    """Dice + BCE + Focal 组合损失，针对小缺陷刷榜"""
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 2.0]  # Focal权重更高，对抗类别不平衡
        self.w_dice, self.w_bce, self.w_focal = weights
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        loss = 0
        if self.w_dice > 0:
            loss += self.w_dice * self.dice(preds, targets)
        if self.w_bce > 0:
            loss += self.w_bce * self.ce(preds, targets)
        if self.w_focal > 0:
            loss += self.w_focal * self.focal(preds, targets)
        return loss


# ================= 指标 =================
def iou_score(pred, target, num_classes=4, ignore_index=0):
    """计算mIoU，可忽略背景"""
    pred = pred.flatten()
    target = target.flatten()
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union == 0:
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)


# ================= 模型加载辅助 =================
def load_model_checkpoint(model, checkpoint_path, strict=True):
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
    return model
