# ─────────────────────────────────────────────
# train.py — Joint Denoising + Inpainting (Auto-Resume, Stable)
# No clean leakage • consistent patch size • corrected upsampling
# ─────────────────────────────────────────────

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import DenoiseInpaintModel
from dataloader import get_dataloader
from config import *

# ─────────────────────────────────────────────
# PatchGAN Discriminator
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# Adversarial loss helper
# ─────────────────────────────────────────────
def compute_adversarial_loss(disc, real, fake, adv_criterion):
    pred_real = disc(real)
    pred_fake = disc(fake.detach())
    loss_D_real = adv_criterion(pred_real, torch.ones_like(pred_real))
    loss_D_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
    loss_D = 0.5 * (loss_D_real + loss_D_fake)

    pred_fake_for_G = disc(fake)
    loss_G_adv = adv_criterion(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
    return loss_D, loss_G_adv

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
def validate(model, fusion_conv, discriminator, dataloader, device,
             mse_loss, l1_loss, adv_criterion):
    model.eval(); fusion_conv.eval(); discriminator.eval()
    val_loss_denoise, val_loss_inpaint = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            noisy = batch["noisy"].to(device)
            masked = batch["masked"].to(device)
            mask = batch["mask"].to(device)
            clean = batch["clean"].to(device)

            # Fuse degraded inputs only (no leakage)
            x_fused = torch.cat([noisy, masked], dim=1)  # [B,6,H,W]
            x_fused = fusion_conv(x_fused)               # [B,3,H,W]

            out = model(x_fused, mask=mask)
            denoised, inpainted = out["denoised"], out["inpainted"]

            loss_denoise = mse_loss(denoised, clean)
            loss_l1 = l1_loss(inpainted, clean)
            pred_fake = discriminator(inpainted)
            loss_adv = adv_criterion(pred_fake, torch.ones_like(pred_fake))
            loss_inpaint = LAMBDA_L1 * loss_l1 + LAMBDA_ADV * loss_adv

            val_loss_denoise += loss_denoise.item() * noisy.size(0)
            val_loss_inpaint += loss_inpaint.item() * noisy.size(0)

    n = len(dataloader.dataset)
    return val_loss_denoise / n, val_loss_inpaint / n

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Initialize models
    model = DenoiseInpaintModel(
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        pretrained=True
    ).to(DEVICE)
    model.enable_lora_training()

    discriminator = PatchDiscriminator().to(DEVICE)
    fusion_conv = nn.Conv2d(6, 3, kernel_size=1).to(DEVICE)

    # Losses
    mse_loss, l1_loss = nn.MSELoss(), nn.L1Loss()
    adv_criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizer_G = Adam(
        list(model.get_trainable_params()) + list(fusion_conv.parameters()),
        lr=LEARNING_RATE
    )
    optimizer_D = Adam(discriminator.parameters(), lr=LR_DISCRIMINATOR)

    # Resume settings
    # ─────────────────────────────────────────────
    # Resume logic — controlled by config flag
    # ─────────────────────────────────────────────
    start_epoch, best_val_loss = 1, float("inf")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    if RESUME_TRAINING and os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming training from: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)

        # Handle both structured and plain model weights
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            fusion_conv.load_state_dict(ckpt.get("fusion_conv", fusion_conv.state_dict()))
            discriminator.load_state_dict(ckpt.get("discriminator", discriminator.state_dict()))
            optimizer_G.load_state_dict(ckpt.get("optimizer_G", optimizer_G.state_dict()))
            optimizer_D.load_state_dict(ckpt.get("optimizer_D", optimizer_D.state_dict()))
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"✅ Resumed full checkpoint (epoch {start_epoch-1}, best val {best_val_loss:.6f})")

        else:
            model.load_state_dict(ckpt, strict=False)
            print("✅ Loaded model weights only — optimizer states not found (fine-tuning mode).")

    elif RESUME_TRAINING:
        print(f"⚠️ Resume flag is True but checkpoint not found: {RESUME_PATH}")
        print("Starting from scratch instead.")
    else:
        print("🆕 RESUME_TRAINING=False — starting from scratch.")


    # Dataloaders
    train_loader = get_dataloader(DATA_ROOT, "train", BATCH_SIZE_TRAIN, IMAGE_SIZE, NUM_WORKERS)
    val_loader = get_dataloader(DATA_ROOT, "validation", BATCH_SIZE_VAL, IMAGE_SIZE, NUM_WORKERS)

    # Logs
    train_losses_denoise, val_losses_denoise = [], []
    train_losses_inpaint, val_losses_inpaint = [], []
    total_train_loss, total_val_loss = [], []

    # ─── Training Loop ───
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train(); fusion_conv.train(); discriminator.train()
        total_loss_denoise, total_loss_inpaint = 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            noisy = batch["noisy"].to(DEVICE)
            masked = batch["masked"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)

            # Fuse degraded inputs
            x_fused = torch.cat([noisy, masked], dim=1)
            x_fused = fusion_conv(x_fused)

            out = model(x_fused, mask=mask)

            denoised, inpainted = out["denoised"], out["inpainted"]

            # Losses
            loss_denoise = mse_loss(denoised, clean)
            loss_l1 = l1_loss(inpainted, clean)
            loss_D, loss_G_adv = compute_adversarial_loss(discriminator, clean, inpainted, adv_criterion)
            loss_inpaint = LAMBDA_L1 * loss_l1 + LAMBDA_ADV * loss_G_adv
            total_G_loss = loss_denoise + loss_inpaint

            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            total_loss_denoise += loss_denoise.item() * noisy.size(0)
            total_loss_inpaint += loss_inpaint.item() * noisy.size(0)

        # Validation
        val_loss_denoise, val_loss_inpaint = validate(
            model, fusion_conv, discriminator, val_loader,
            DEVICE, mse_loss, l1_loss, adv_criterion
        )

        # Average losses
        avg_train_denoise = total_loss_denoise / len(train_loader.dataset)
        avg_train_inpaint = total_loss_inpaint / len(train_loader.dataset)
        avg_train_total = avg_train_denoise + avg_train_inpaint
        avg_val_total = val_loss_denoise + val_loss_inpaint

        # Track losses
        train_losses_denoise.append(avg_train_denoise)
        val_losses_denoise.append(val_loss_denoise)
        train_losses_inpaint.append(avg_train_inpaint)
        val_losses_inpaint.append(val_loss_inpaint)
        total_train_loss.append(avg_train_total)
        total_val_loss.append(avg_val_total)

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print(f" Denoise → Train: {avg_train_denoise:.6f}, Val: {val_loss_denoise:.6f}")
        print(f" Inpaint → Train: {avg_train_inpaint:.6f}, Val: {val_loss_inpaint:.6f}")
        print(f" Total → Train: {avg_train_total:.6f}, Val: {avg_val_total:.6f}")

        # Save best model
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "best_model.pth"))
            print(f" ✅ Best model updated at epoch {epoch} (val loss = {best_val_loss:.6f})")

        # Save full checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "fusion_conv": fusion_conv.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(ckpt, RESUME_PATH)

    # ─── Plot losses ───
    plt.figure(figsize=(8,5))
    plt.plot(train_losses_denoise, label="Train Denoise (MSE)")
    plt.plot(val_losses_denoise, label="Val Denoise (MSE)", linestyle="--")
    plt.plot(train_losses_inpaint, label="Train Inpaint (L1+Adv)")
    plt.plot(val_losses_inpaint, label="Val Inpaint (L1)", linestyle="--")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "separate_loss_plot.png")); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(total_train_loss, label="Total Train Loss")
    plt.plot(total_val_loss, label="Total Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "loss_plot.png")); plt.close()

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
