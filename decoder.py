# decoder.py
import torch
import torch.nn as nn
import math


def reshape_vit_features_to_map(features, image_size, patch_size):
    """
    Convert ViT patch embeddings [B, N, C] → feature map [B, C, H', W'].
    """
    B, N, C = features.shape
    h = w = int(math.sqrt(N))
    feat_map = features.permute(0, 2, 1).reshape(B, C, h, w)
    return feat_map


class ConvUpsampleBlock(nn.Module):
    """A basic upsampling + conv + norm block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DenoisingDecoder(nn.Module):
    """
    Decoder for denoising task.
    Takes encoder patch embeddings and reconstructs the clean image.
    """
    def __init__(self, embed_dim=768, image_size=128, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # feature map shape after reshaping
        self.base_h = self.image_size // self.patch_size

        # decoder layers
        self.decoder = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),
            ConvUpsampleBlock(512, 256),
            ConvUpsampleBlock(256, 128),
            ConvUpsampleBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # normalize output 0–1
        )

    def forward(self, features):
        """
        Args:
            features: [B, N, C] ViT embeddings
        Returns:
            denoised reconstruction [B, 3, H, W]
        """
        feat_map = reshape_vit_features_to_map(features, self.image_size, self.patch_size)
        output = self.decoder(feat_map)
        return output


class InpaintingDecoder(nn.Module):
    """
    Decoder for inpainting task.
    Uses encoder patch embeddings and mask to reconstruct missing regions.
    """
    def __init__(self, embed_dim=768, image_size=128, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # basic upsampling path
        self.up_path = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),
            ConvUpsampleBlock(512, 256),
            ConvUpsampleBlock(256, 128),
            ConvUpsampleBlock(128, 64)
        )

        # merge mask + feature map for guided reconstruction
        self.merge = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features, mask):
        """
        Args:
            features: [B, N, C] from encoder
            mask: [B, 3, H, W] or [B, 1, H, W]
        Returns:
            inpainted image [B, 3, H, W]
        """
        feat_map = reshape_vit_features_to_map(features, self.image_size, self.patch_size)
        feat = self.up_path(feat_map)

        if mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]  # use single channel mask

        # Resize mask to decoder feature map size
        mask_resized = torch.nn.functional.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([feat, mask_resized], dim=1)

        output = self.merge(merged)
        return output


class DualTaskDecoder(nn.Module):
    """
    Combines both decoders for multi-task use.
    """
    def __init__(self, embed_dim=768, image_size=128, patch_size=16):
        super().__init__()
        self.denoise_decoder = DenoisingDecoder(embed_dim, image_size, patch_size)
        self.inpaint_decoder = InpaintingDecoder(embed_dim, image_size, patch_size)

    def forward(self, features, mask=None):
        """
        Returns:
            denoised_img, inpainted_img
        """
        denoised = self.denoise_decoder(features)
        inpainted = self.inpaint_decoder(features, mask) if mask is not None else None
        return denoised, inpainted
