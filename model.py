# ─────────────────────────────────────────────
# model.py — Joint Denoising + Inpainting (Fused Input)
# Encoder receives a single fused input image
# Decoder jointly reconstructs denoised & inpainted outputs
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
from encoder import DinoV3SmallEncoder
from decoder import DualTaskDecoder


class DenoiseInpaintModel(nn.Module):
    """
    End-to-end model combining:
    - Pretrained DINOv3-Small encoder (with LoRA adapters)
    - Dual-task decoder for denoising & inpainting
    The encoder processes a fused input (noisy + masked projected to 3 channels).
    """

    def __init__(
        self,
        model_name="vit_small_patch16_dinov3",  # ✅ Correct model name
        image_size=128,
        patch_size=16,                          # ✅ Match DINOv3-small patch size
        embed_dim=384,                          # ✅ Match DINOv3-small output dim
        lora_r=8,
        lora_alpha=16,
        pretrained=True,
    ):
        super().__init__()

        # Shared DINOv3-Small Encoder with LoRA adapters
        self.encoder = DinoV3SmallEncoder(
            model_name=model_name,
            pretrained=pretrained,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            image_size=image_size,
        )

        # Dual-task Decoder (for denoising + inpainting)
        self.decoder = DualTaskDecoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
        )

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────
    def forward(self, x_fused, mask=None):
        """
        Args:
            x_fused: [B, 3, H, W] — fused image (noisy+masked projected to 3 ch)
            mask:    [B, 1, H, W] — binary inpainting mask (optional)
        Returns:
            dict with:
                "features":   encoder patch embeddings
                "denoised":   reconstructed clean image
                "inpainted":  reconstructed masked regions
        """
        features = self.encoder(x_fused)  # [B, N, 384]
        denoised, inpainted = self.decoder(features, mask)
        return {
            "features": features,
            "denoised": denoised,
            "inpainted": inpainted,
        }

    # ─────────────────────────────────────────────
    # Training Utilities
    # ─────────────────────────────────────────────
    def enable_lora_training(self):
        """Enable training for LoRA adapters + decoder, freeze the rest."""
        for name, p in self.encoder.named_parameters():
            p.requires_grad = "lora" in name.lower()
        for p in self.decoder.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        """Return trainable parameters (LoRA + decoder)."""
        return [p for p in self.parameters() if p.requires_grad]
