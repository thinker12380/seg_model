import torch
import torch.nn as nn
import torch.nn.functional as F
from models_official import SegNeXt_Official, Segmenter_Official_Linear, Segmenter_Official_Mask


# ================= 简化版 MSCA 模块 =================
class MSCA(nn.Module):
    """
    Multi-Scale Convolutional Attention (简化版)
    基于SegNeXt文章的MSCA设计，适配刷榜需求
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 局部聚合
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # 多尺度dilated depth-wise（使用strip近似思想，降低计算量）
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)
        self.dil1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dil2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        # 1x1 注意力生成
        self.attn = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        local = self.local_conv(x)
        s1 = self.dw1(x)
        s2 = self.dw2(x)
        d1 = self.dil1(x)
        d2 = self.dil2(x)
        multi = s1 + s2 + d1 + d2  # 多尺度聚合
        attn = self.attn(torch.cat([local, multi, d1, d2], dim=1))
        return x * attn + x  # 残差连接


class MSCAStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth=3, downsample=True):
        super().__init__()
        if downsample:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.blocks = nn.Sequential(*[MSCA(out_ch) for _ in range(depth)])

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class SegNeXtEncoder(nn.Module):
    """轻量版SegNeXt Encoder，金字塔结构"""
    def __init__(self, in_ch=3, embed_dims=[32, 64, 128, 256], depths=[2, 2, 4, 2]):
        super().__init__()
        self.stages = nn.ModuleList()
        for i, (ed, dp) in enumerate(zip(embed_dims, depths)):
            in_c = in_ch if i == 0 else embed_dims[i - 1]
            self.stages.append(MSCAStage(in_c, ed, depth=dp, downsample=(i > 0)))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class HambergerDecoder(nn.Module):
    """
    简化版 Hamberger 全局建模模块
    使用矩阵分解思想（低秩近似）替代复杂自注意力
    """
    def __init__(self, in_dims=[64, 128, 256], out_dim=128, num_classes=4):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(d, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        ) for d in in_dims])
        # 低秩全局建模
        self.rank = 16
        self.U = nn.Parameter(torch.randn(out_dim, self.rank))
        self.S = nn.Parameter(torch.randn(self.rank, self.rank))
        self.V = nn.Parameter(torch.randn(self.rank, out_dim))
        self.bn = nn.BatchNorm2d(out_dim)
        self.num_classes = num_classes
        self.seg_head = nn.Conv2d(out_dim, num_classes, 1)

    def forward(self, features):
        # features: [stage2, stage3, stage4] (忽略stage1)
        target_size = features[0].shape[2:]
        outs = []
        for i, f in enumerate(features):
            x = self.convs[i](f)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)
        x = sum(outs)  # 逐元素相加融合
        B, C, H, W = x.shape
        # 低秩近似全局建模
        x_flat = x.view(B, C, -1)  # [B, C, HW]
        # 低秩全局建模: proj = U^T @ x, proj = S @ proj, x_global = V @ proj
        proj = torch.einsum("bcn,cr->brn", x_flat, self.U)      # [B, rank, HW]
        proj = torch.einsum("brn,rs->bsn", proj, self.S)        # [B, rank, HW]
        x_global = torch.einsum("bsn,sc->bcn", proj, self.V)    # [B, C, HW]
        x_global = x_global.view(B, C, H, W)
        x = x + self.bn(x_global)
        x = self.seg_head(x)
        return x


class SegNeXtLite(nn.Module):
    """轻量SegNeXt，用于刷榜实验"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = SegNeXtEncoder(embed_dims=[32, 64, 128, 256], depths=[2, 2, 4, 2])
        self.decoder = HambergerDecoder(in_dims=[64, 128, 256], out_dim=128, num_classes=num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        # 只取后三个stage
        out = self.decoder(feats[1:])
        # 上采样回原图尺寸
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# ================= Vision Transformer (ViT) 分割模型 =================
class PatchEmbed(nn.Module):
    """
    Patch Embedding: 将输入图像分割为固定大小的patch，再线性投影为向量。
    例如 224x224 图像, patch=16x16, 则生成 (224/16)^2 = 196 个patch,
    每个patch维度 16x16x3=768, 输出 [B, 196, 768]。
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, embed_dim, H//ps, W//ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention (标准 Transformer)"""
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Transformer 中的 Feed Forward Network (MLP)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """标准 ViT Block: LN -> Multi-Head Attention -> LN -> MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """
    Vision Transformer 骨干网络。
    为适配分割任务，不使用显式可学习位置编码（ViT实验表明对性能影响有限），
    从而天然支持任意输入分辨率。
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # [B, N+1, embed_dim]


class ViTSegmentation(nn.Module):
    """
    ViT 分割模型: ViT Encoder + 轻量 CNN Decoder。
    支持任意分辨率输入（H, W 需为 patch_size 整数倍）。
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 num_classes=4):
        super().__init__()
        self.encoder = ViTBackbone(patch_size, in_chans, embed_dim, depth, num_heads)
        self.patch_size = patch_size
        # Decoder: 逐步上采样
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.encoder(x)          # [B, N+1, embed_dim]
        feat = feat[:, 1:]              # 去掉 cls token
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        feat = feat.transpose(1, 2).reshape(B, -1, grid_h, grid_w)
        out = self.decoder(feat)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ================= 使用 segmentation-models-pytorch 构建强基线 =================
def create_smp_model(model_name="unetplusplus", encoder_name="tu-tf_efficientnetv2_l", num_classes=4, pretrained=False):
    """
    创建SMP模型，推荐刷榜配置：
    - model_name: unetplusplus / deeplabv3plus / unet / fpn / manet
    - encoder_name: tu-tf_efficientnetv2_l / tu-maxvit_base_tf_512 / tu-coatnet_2_rw_224
    - pretrained: 是否下载ImageNet预训练权重（无网络时设为False）
    """
    import segmentation_models_pytorch as smp
    model = smp.create_model(
        arch=model_name,
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    return model


# ================= 模型封装入口 =================
MODEL_REGISTRY = {
    "segnext_lite": SegNeXtLite,
    "segnext_official": SegNeXt_Official,
    "segmenter_linear": lambda nc: Segmenter_Official_Linear(n_cls=nc),
    "segmenter_mask": lambda nc: Segmenter_Official_Mask(n_cls=nc),
    "unet": lambda nc: create_smp_model("unet", "tu-tf_efficientnetv2_l", nc, pretrained=False),
    "unetplusplus": lambda nc: create_smp_model("unetplusplus", "tu-tf_efficientnetv2_l", nc, pretrained=False),
    "deeplabv3plus": lambda nc: create_smp_model("deeplabv3plus", "tu-tf_efficientnetv2_l", nc, pretrained=False),
    "fpn": lambda nc: create_smp_model("fpn", "tu-tf_efficientnetv2_l", nc, pretrained=False),
    "unetplusplus_maxvit": lambda nc: create_smp_model("unetplusplus", "tu-maxvit_base_tf_512", nc, pretrained=False),
    "vit_base": lambda nc: ViTSegmentation(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=nc),
    "vit_small": lambda nc: ViTSegmentation(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=nc),
}


def build_model(name="unetplusplus", num_classes=4):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes)
