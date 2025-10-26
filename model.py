# model.py
import torch
import torch.nn as nn
from encoder import DinoVitEncoder
from decoder import DualTaskDecoder


class DenoiseInpaintModel(nn.Module):
    """
    End-to-end model combining:
    - Pretrained frozen DINO-ViT encoder with LoRA adapters
    - Dual decoders for denoising and inpainting tasks
    """

    def __init__(self, 
                 model_name="vit_base_patch16_224.dino", 
                 image_size=128, 
                 patch_size=16,
                 embed_dim=768,
                 lora_r=8, 
                 lora_alpha=16,
                 pretrained=True):
        super().__init__()

        # Encoder (frozen DINO-ViT + LoRA)
        self.encoder = DinoVitEncoder(
            model_name=model_name,
            pretrained=pretrained,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # Decoder (dual-task: denoising + inpainting)
        self.decoder = DualTaskDecoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size
        )

    def forward(self, x, mask=None):
        """
        Forward pass through encoder and both decoders.
        Args:
            x: Input image tensor [B, 3, H, W]
            mask: Optional mask tensor [B, 1 or 3, H, W] for inpainting
        Returns:
            {
              "features": ViT latent embeddings [B, N, C],
              "denoised": reconstructed clean image [B, 3, H, W],
              "inpainted": reconstructed masked image [B, 3, H, W]
            }
        """
        features = self.encoder(x)
        denoised, inpainted = self.decoder(features, mask)

        return {
            "features": features,
            "denoised": denoised,
            "inpainted": inpainted
        }

    def enable_lora_training(self):
        """Enable training only for LoRA adapter and decoder parameters."""
        # Disable all encoder backbone params
        for name, p in self.encoder.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Enable decoder params
        for p in self.decoder.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        """Return trainable parameters (LoRA + decoder)."""
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
        return params
