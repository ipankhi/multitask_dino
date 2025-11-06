# ─────────────────────────────────────────────
# decoder.py — Multi-Task Decoder (Denoising + Inpainting)
# For vit_small_patch16_dinov3 (embed_dim=384, patch_size=16)
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Reshape helper
# ─────────────────────────────────────────────
def reshape_vit_features_to_map(features, image_size, patch_size):
    """
    Convert ViT patch embeddings [B, N, C] → feature map [B, C, H', W'].
    Handles DINOv3 outputs (with CLS + register tokens).
    """
    B, N, C = features.shape

    # Expected spatial tokens (e.g., 8×8=64 for 128×128, patch16)
    grid_h = grid_w = image_size // patch_size
    N_expected = grid_h * grid_w

    # Drop CLS + extra tokens if present
    if N > N_expected:
        features = features[:, -N_expected:, :]  # keep only spatial tokens

    feat_map = features.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)
    return feat_map


# ─────────────────────────────────────────────
# Core upsampling block
# ─────────────────────────────────────────────
class ConvUpsampleBlock(nn.Module):
    """Upsample + ConvTranspose2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Denoising Decoder
# ─────────────────────────────────────────────
class DenoisingDecoder(nn.Module):
    def __init__(self, embed_dim=384, image_size=128, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        self.decoder = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),
            ConvUpsampleBlock(512, 256),
            ConvUpsampleBlock(256, 128),
            ConvUpsampleBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        feat_map = reshape_vit_features_to_map(features, self.image_size, self.patch_size)
        return self.decoder(feat_map)


# ─────────────────────────────────────────────
# Inpainting Decoder
# ─────────────────────────────────────────────
class InpaintingDecoder(nn.Module):
    def __init__(self, embed_dim=384, image_size=128, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # Upsampling path
        self.up_path = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),
            ConvUpsampleBlock(512, 256),
            ConvUpsampleBlock(256, 128),
            ConvUpsampleBlock(128, 64),
        )

        # Mask-guided reconstruction
        self.merge = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features, mask):
        feat_map = reshape_vit_features_to_map(features, self.image_size, self.patch_size)
        feat = self.up_path(feat_map)

        if mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]  # keep only one mask channel

        mask_resized = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([feat, mask_resized], dim=1)
        return self.merge(merged)


# ─────────────────────────────────────────────
# Dual-task Decoder (Combines both tasks)
# ─────────────────────────────────────────────
class DualTaskDecoder(nn.Module):
    def __init__(self, embed_dim=384, image_size=128, patch_size=16):
        super().__init__()
        self.denoise_decoder = DenoisingDecoder(embed_dim, image_size, patch_size)
        self.inpaint_decoder = InpaintingDecoder(embed_dim, image_size, patch_size)

    def forward(self, features, mask=None):
        denoised = self.denoise_decoder(features)
        inpainted = self.inpaint_decoder(features, mask) if mask is not None else None
        return denoised, inpainted
