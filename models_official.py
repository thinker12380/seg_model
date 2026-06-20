"""
官方源码精华集成
- SegNeXt (MMSegmentation): 精确 MSCAN + LightHamHead
- Segmenter (ICCV 2021): ViT + DecoderLinear / MaskTransformer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== DropPath / Layer Scale 工具 ========================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    """Rescale position embeddings to different grid size."""
    posemb_tok, posemb_grid = posemb[:, :num_extra_tokens], posemb[0, num_extra_tokens:]
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape
    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


# ======================== SegNeXt 官方实现 ========================
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        return self.dwconv(x)


class Mlp_Official(nn.Module):
    """官方 Mlp，含 DWConv"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv_Official(nn.Module):
    """官方 Stem：两层 3x3 Conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule_Official(nn.Module):
    """官方 MSCA：多尺度条带卷积注意力"""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class SpatialAttention_Official(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule_Official(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class SegNeXtBlock_Official(nn.Module):
    """官方 Block：LN + MSCA + Layer Scale + DropPath + Mlp"""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention_Official(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_Official(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed_Official(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN_Official(nn.Module):
    """官方 MSCAN Backbone"""
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0.,
                 depths=[3, 4, 6, 3], num_stages=4):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv_Official(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed_Official(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])
            block = nn.ModuleList([
                SegNeXtBlock_Official(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                      drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class NMF2D_Official(nn.Module):
    """官方 NMF2D (非负矩阵分解)"""
    def __init__(self, args=dict()):
        super().__init__()
        self.S = args.get('S', 1)
        self.R = args.get('R', 64)
        self.train_steps = args.get('train_steps', 6)
        self.eval_steps = args.get('eval_steps', 7)
        self.inv_t = args.get('inv_t', 25)
        self.rand_init = args.get('rand_init', True)

        self._lambda = nn.Parameter(torch.ones(1) * self.inv_t, requires_grad=True)

        if self.rand_init:
            self.ccoef = nn.Parameter(torch.ones(1), requires_grad=True)
            self.cbase = nn.Parameter(torch.ones(1), requires_grad=True)

    def _build_bases(self, B, S, D, R, cuda=False):
        if not self.rand_init:
            bases = torch.ones((B * S, D, R)).to('cuda' if cuda else 'cpu')
        else:
            bases = self.cbase * torch.randn((B * S, D, R)).to('cuda' if cuda else 'cpu')
        bases = F.relu(bases)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape
        S, D, R = self.S, C // self.S, self.R
        x_3d = x.view(B * S, D, H * W)
        if not self.rand_init:
            self.ccoef = 1.0
        coef = self.ccoef * torch.randn((B * S, H * W, R)).to(x.device)
        coef = F.relu(coef)
        bases = self._build_bases(B, S, D, R, cuda=x.is_cuda)
        for _ in range(self.train_steps if self.training else self.eval_steps):
            bases, coef = self.local_step(x_3d, bases, coef)
            coef = self.compute_coef(x_3d, bases, coef)
        bases = F.relu(bases)
        # 重建: (B*S, D, R) @ (B*S, R, H*W) -> (B*S, D, H*W)
        x_hat = torch.bmm(bases, coef.transpose(1, 2))
        x_hat = x_hat.view(B, C, H, W)
        return x_hat


class Hamburger_Official(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict()):
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1)
        self.ham = NMF2D_Official(ham_kwargs)
        self.ham_out = nn.Conv2d(ham_channels, ham_channels, 1)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


class LightHamHead_Official(nn.Module):
    """官方 LightHamHead Decoder"""
    def __init__(self, in_channels, num_classes=4, ham_channels=512, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.squeeze = nn.Sequential(
            nn.Conv2d(sum(in_channels), ham_channels, 1),
            nn.BatchNorm2d(ham_channels),
            nn.ReLU(inplace=True))
        self.hamburger = Hamburger_Official(ham_channels)
        self.align = nn.Sequential(
            nn.Conv2d(ham_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.cls_seg = nn.Conv2d(256, num_classes, 1)

    def forward(self, inputs):
        inputs = [F.interpolate(level, size=inputs[0].shape[2:], mode='bilinear',
                                align_corners=self.align_corners) for level in inputs]
        x = torch.cat(inputs, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        return self.cls_seg(x)


class SegNeXt_Official(nn.Module):
    """官方 SegNeXt 完整模型 (MSCAN + LightHamHead)"""
    def __init__(self, num_classes=4, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], depths=[3, 4, 6, 3], drop_path_rate=0.1):
        super().__init__()
        self.encoder = MSCAN_Official(embed_dims=embed_dims, mlp_ratios=mlp_ratios,
                                      depths=depths, drop_path_rate=drop_path_rate)
        self.decoder = LightHamHead_Official(in_channels=embed_dims, num_classes=num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# ======================== Segmenter 官方实现 ========================
class PatchEmbedding_Official(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels=3):
        super().__init__()
        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention_Official(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block_Official(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention_Official(dim, heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(self.norm1(x), mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer_Official(nn.Module):
    """官方 ViT Backbone (含 pos_embed, DropPath)"""
    def __init__(self, image_size, patch_size, n_layers, d_model, d_ff, n_heads,
                 n_cls=0, dropout=0.1, drop_path_rate=0.0, distilled=False, channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding_Official(image_size, patch_size, d_model, channels)
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.distilled = distilled
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 2, d_model))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([Block_Official(d_model, n_heads, d_ff, dropout, dpr[i])
                                     for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if return_features:
            return x
        return x[:, 0] if self.n_cls > 0 else x


class DecoderLinear_Official(nn.Module):
    """官方 DecoderLinear：极简，仅一个 Linear"""
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.head = nn.Linear(d_encoder, n_cls)
        self.apply(init_weights)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        B, N, C = x.shape
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, self.n_cls, GS, GS)
        return x


class MaskTransformer_Official(nn.Module):
    """官方 MaskTransformer Decoder"""
    def __init__(self, n_cls, patch_size, d_encoder, n_layers, n_heads, d_model, d_ff,
                 drop_path_rate, dropout):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([Block_Official(d_model, n_heads, d_ff, dropout, dpr[i])
                                     for i in range(n_layers)])
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        B = masks.shape[0]
        masks = masks.reshape(B, GS, GS, self.n_cls).permute(0, 3, 1, 2)
        return masks


def padding(img, patch_size):
    """官方 padding：使尺寸为 patch_size 倍数"""
    _, _, H, W = img.shape
    pad_h, pad_w = 0, 0
    if H % patch_size != 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size != 0:
        pad_w = patch_size - (W % patch_size)
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img


def unpadding(img, target_size):
    """官方 unpadding：裁剪回原始尺寸"""
    _, _, H, W = img.shape
    h, w = target_size
    if H > h:
        img = img[:, :, :h, :]
    if W > w:
        img = img[:, :, :, :w]
    return img


class Segmenter_Official_Linear(nn.Module):
    """官方 Segmenter + Linear Decoder"""
    def __init__(self, image_size=(512, 512), patch_size=16, n_layers=12, d_model=384,
                 d_ff=1536, n_heads=6, n_cls=4, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.encoder = VisionTransformer_Official(image_size, patch_size, n_layers, d_model,
                                                   d_ff, n_heads, dropout=dropout,
                                                   drop_path_rate=drop_path_rate)
        self.decoder = DecoderLinear_Official(n_cls, patch_size, d_model)

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        x = self.encoder(im, return_features=True)
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))
        return masks


class Segmenter_Official_Mask(nn.Module):
    """官方 Segmenter + MaskTransformer Decoder"""
    def __init__(self, image_size=(512, 512), patch_size=16, n_layers=12, d_model=384,
                 d_ff=1536, n_heads=6, n_cls=4, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.encoder = VisionTransformer_Official(image_size, patch_size, n_layers, d_model,
                                                   d_ff, n_heads, dropout=dropout,
                                                   drop_path_rate=drop_path_rate)
        self.decoder = MaskTransformer_Official(n_cls, patch_size, d_model, n_layers=2,
                                                 n_heads=6, d_model=d_model, d_ff=d_ff,
                                                 drop_path_rate=0.0, dropout=dropout)

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        x = self.encoder(im, return_features=True)
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))
        return masks
