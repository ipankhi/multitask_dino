# encoder.py
import torch
import torch.nn as nn
import timm
import math
from lora import inject_lora_to_last_block


class DinoVitEncoder(nn.Module):
    """
    Encoder built from a pretrained DINO-ViT backbone (via timm),
    with frozen weights and LoRA adapters injected into the last transformer block.
    Supports flexible input sizes (128x128, 224x224, etc.).
    """

    def __init__(self, model_name="vit_base_patch16_224.dino", pretrained=True,
                 image_size=128, lora_r=8, lora_alpha=16):
        super().__init__()

        self.image_size = image_size

        # --- Load pretrained DINO-ViT backbone ---
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,     # remove classification head
            global_pool=""     # keep patch embeddings
        )
        print(f"[Encoder] Loaded pretrained backbone: {model_name}")

        # --- Freeze all pretrained weights ---
        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Adjust patch embedding layer for custom image size ---
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "img_size"):
            self.backbone.patch_embed.img_size = (image_size, image_size)

        # --- Interpolate positional embeddings to match new resolution ---
        if hasattr(self.backbone, "pos_embed"):
            pos_embed = self.backbone.pos_embed  # shape: [1, num_patches+1, dim]
            if pos_embed is not None:
                num_patches_old = pos_embed.shape[1] - 1
                dim = pos_embed.shape[-1]
                h_old = w_old = int(math.sqrt(num_patches_old))

                # Extract class token + patch grid
                cls_token = pos_embed[:, 0:1, :]
                grid = pos_embed[:, 1:, :].reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)

                # --- Safely get patch size (handles tuple or int) ---
                patch_size = self.backbone.patch_embed.patch_size
                if isinstance(patch_size, tuple):
                    patch_size = patch_size[0]

                # Compute new patch grid size
                h_new = w_new = image_size // patch_size

                # Interpolate position embeddings
                grid_resized = torch.nn.functional.interpolate(
                    grid,
                    size=(h_new, w_new),
                    mode='bicubic',
                    align_corners=False
                )

                # Recombine cls token + resized grid
                grid_resized = grid_resized.permute(0, 2, 3, 1).reshape(1, h_new * w_new, dim)
                new_pos_embed = torch.cat((cls_token, grid_resized), dim=1)
                self.backbone.pos_embed = nn.Parameter(new_pos_embed)

                #print(f"[Encoder] Positional embeddings resized from {h_old}x{w_old} → {h_new}x{w_new}")

        # --- Inject LoRA adapters into the last transformer block ---
        wrapped = inject_lora_to_last_block(self.backbone, r=lora_r, alpha=lora_alpha)
        print(f"[LoRA] Applied to layers: {wrapped}")

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    def forward(self, x):
        """
        Forward through the frozen ViT backbone.
        Returns patch embeddings (without classification head).
        """
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        # Remove CLS token if present
        if feats.ndim == 3 and feats.size(1) > 1:
            feats = feats[:, 1:, :]  # [B, N, C]

        return feats

    # ---------------------------------------------------
    # Utility methods
    # ---------------------------------------------------
    def lora_parameters(self):
        """Return only LoRA adapter parameters for training."""
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                yield param

    def enable_lora_training(self):
        """Enable only LoRA parameters for training."""
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False


# ---------------------------------------------------
# Test Example
# ---------------------------------------------------
# if __name__ == "__main__":
#     model = DINOvITEncoder(
#         model_name="vit_base_patch16_224.dino",
#         pretrained=True,
#         image_size=128,
#         lora_r=8,
#         lora_alpha=16
#     )

#     x = torch.randn(2, 3, 128, 128)
#     with torch.no_grad():
#         feats = model(x)

#     print("Feature shape:", feats.shape)
