import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding as per ViT paper
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
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
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Encoder Block as per the paper"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """Decoder block for TransUNet"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_proj(skip)

        # Handle size mismatch between skip and upsampled feature
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    """
    TransUNet model implementation exactly following the paper:
    """
    def __init__(self, img_dim=224, patch_dim=16, num_channels=3, num_classes=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.0):
        super().__init__()
        resnet = models.resnet50(weights=None)
        # resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 64 channels, 1/4 resolution
        self.encoder2 = resnet.layer1  # 256 channels, 1/4 resolution
        self.encoder3 = resnet.layer2  # 512 channels, 1/8 resolution
        self.encoder4 = resnet.layer3  # 1024 channels, 1/16 resolution
        self.encoder5 = resnet.layer4  # 2048 channels, 1/32 resolution

        self.patch_size = patch_dim
        self.flatten_dim = (img_dim // 32) ** 2  # Based on ResNet output size (1/32 of input)
        self.linear_encoding = nn.Linear(2048, embed_dim)  # Map CNN features to Transformer dimension

        self.pos_embed = nn.Parameter(torch.zeros(1, self.flatten_dim, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.conv_proj = nn.Conv2d(embed_dim, 512, kernel_size=1)  # As per paper Fig 1

        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=1024, out_channels=256)  # 512→256, 1/16→1/8
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=512, out_channels=128)   # 256→128, 1/8→1/4
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=256, out_channels=64)    # 128→64, 1/4→1/2
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=16)      # 64→16, 1/2→1/1

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        orig_img = x  # Store for skip to decoder1
        e1 = self.encoder1(x)      # 1/4 resolution
        e2 = self.encoder2(e1)     # 1/4 resolution
        e3 = self.encoder3(e2)     # 1/8 resolution
        e4 = self.encoder4(e3)     # 1/16 resolution
        e5 = self.encoder5(e4)     # 1/32 resolution, 2048 channels

        B, C, H, W = e5.shape

        x_flat = e5.flatten(2).transpose(1, 2)  # (B, H*W, 2048)
        x_tr = self.linear_encoding(x_flat)  # (B, H*W, embed_dim)
        x_tr = x_tr + self.pos_embed

        for blk in self.transformer_blocks:
            x_tr = blk(x_tr)

        x_unflat = x_tr.transpose(1, 2).view(B, -1, H, W)  # (B, embed_dim, H, W)

        x_proj = self.conv_proj(x_unflat)  # (B, 512, H, W)

        d4 = self.decoder4(x_proj, e4)  # 1/16 resolution
        d3 = self.decoder3(d4, e3)      # 1/8 resolution
        d2 = self.decoder2(d3, e2)      # 1/4 resolution

        x_skip = F.interpolate(orig_img, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(d2, e1)      # Full resolution

        out = self.final_conv(d1)

        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        return torch.sigmoid(out)
