# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import DenoiseInpaintModel
from dataset import get_dataloader
from config import *

# ─────────────────────────────────────────────
# Adversarial Discriminator
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator for inpainting adversarial loss."""
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

            nn.Conv2d(base_channels * 4, 1, 4, 1, 0)  # output patch-level real/fake map
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────
def compute_adversarial_loss(disc, real, fake, adv_criterion):
    """Compute adversarial loss for generator & discriminator."""
    # Discriminator loss
    pred_real = disc(real)
    pred_fake = disc(fake.detach())

    loss_D_real = adv_criterion(pred_real, torch.ones_like(pred_real))
    loss_D_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
    loss_D = (loss_D_real + loss_D_fake) / 2

    # Generator loss
    pred_fake_for_G = disc(fake)
    loss_G_adv = adv_criterion(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

    return loss_D, loss_G_adv


# def validate(model, dataloader, device, mse_loss, l1_loss):
#     """Compute average validation loss."""
#     model.eval()
#     val_loss_denoise, val_loss_inpaint = 0.0, 0.0
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Validating"):
#             noisy = batch["noisy"].to(device)
#             clean = batch["clean"].to(device)
#             mask = batch["mask"].to(device)

#             out = model(noisy, mask)
#             denoised, inpainted = out["denoised"], out["inpainted"]

#             loss_denoise = mse_loss(denoised, clean)
#             loss_inpaint = l1_loss(inpainted, clean)
#             val_loss_denoise += loss_denoise.item() * noisy.size(0)
#             val_loss_inpaint += loss_inpaint.item() * noisy.size(0)

#     val_loss_denoise /= len(dataloader.dataset)
#     val_loss_inpaint /= len(dataloader.dataset)
#     return val_loss_denoise, val_loss_inpaint

def validate(model, discriminator, dataloader, device, mse_loss, l1_loss, adv_criterion):
    model.eval()
    discriminator.eval()
    val_loss_denoise, val_loss_inpaint, val_loss_adv = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            mask = batch["mask"].to(device)

            out = model(noisy, mask)
            denoised, inpainted = out["denoised"], out["inpainted"]

            # Denoising Loss (MSE)
            loss_denoise = mse_loss(denoised, clean)

            # Inpainting Loss (L1 + Adversarial)
            loss_l1 = l1_loss(inpainted, clean)
            pred_fake = discriminator(inpainted)
            loss_adv = adv_criterion(pred_fake, torch.ones_like(pred_fake))

            loss_inpaint = LAMBDA_L1 * loss_l1 + LAMBDA_ADV * loss_adv

            val_loss_denoise += loss_denoise.item() * noisy.size(0)
            val_loss_inpaint += loss_inpaint.item() * noisy.size(0)
            #val_loss_adv += loss_adv.item() * noisy.size(0)

    n = len(dataloader.dataset)
    val_loss_denoise /= n
    val_loss_inpaint /= n
    #val_loss_adv /= n

    return val_loss_denoise, val_loss_inpaint

# ─────────────────────────────────────────────
# Main Training Script
# ─────────────────────────────────────────────
def main():
   
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

    # --- Discriminator (for adversarial loss) ---
    discriminator = PatchDiscriminator().to(DEVICE)

    # --- Losses ---
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    adv_criterion = nn.BCEWithLogitsLoss()

    # --- Optimizers ---
    optimizer_G = Adam(model.get_trainable_params(), lr=LEARNING_RATE)
    optimizer_D = Adam(discriminator.parameters(), lr=LR_DISCRIMINATOR)

    # --- Data ---
    train_loader = get_dataloader(DATA_ROOT, split="train", batch_size=BATCH_SIZE_TRAIN, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO, num_patches=NUM_PATCHES)
    val_loader = get_dataloader(DATA_ROOT, split="validation", batch_size=BATCH_SIZE_VAL, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO, num_patches=NUM_PATCHES)

    # --- Tracking losses ---
    train_losses_denoise, val_losses_denoise = [], []
    train_losses_inpaint, val_losses_inpaint = [], []
    train_loss, val_loss = [], []

    best_val_loss = float("inf")
    best_epoch = 0
    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        discriminator.train()

        total_loss_denoise, total_loss_inpaint = 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            noisy = batch["noisy"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            # --- Forward ---
            out = model(noisy, mask)
            denoised, inpainted = out["denoised"], out["inpainted"]

            # --- Denoising Loss (MSE) ---
            loss_denoise = mse_loss(denoised, clean)

            # --- Inpainting Loss (L1 + Adversarial) ---
            loss_l1 = l1_loss(inpainted, clean)
            loss_D, loss_G_adv = compute_adversarial_loss(discriminator, clean, inpainted, adv_criterion)
            loss_inpaint = LAMBDA_L1 * loss_l1 + LAMBDA_ADV  * loss_G_adv  

            # --- Total Generator Loss ---
            total_G_loss = loss_denoise + loss_inpaint

            # --- Backprop Generator ---
            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

            # --- Update Discriminator ---
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            total_loss_denoise += loss_denoise.item() * noisy.size(0)
            total_loss_inpaint += loss_inpaint.item() * noisy.size(0)

        # --- Validation ---
        val_loss_denoise, val_loss_inpaint = validate(model, discriminator, val_loader, DEVICE, mse_loss, l1_loss, adv_criterion)

        # --- Compute Average ---
        avg_train_denoise = total_loss_denoise / len(train_loader.dataset)
        avg_train_inpaint = total_loss_inpaint / len(train_loader.dataset)

        train_losses_denoise.append(avg_train_denoise)
        val_losses_denoise.append(val_loss_denoise)
        train_losses_inpaint.append(avg_train_inpaint)
        val_losses_inpaint.append(val_loss_inpaint)
        
        total_train_loss = avg_train_denoise + avg_train_inpaint
        total_val_loss = val_loss_denoise + val_loss_inpaint
        train_loss.append(total_train_loss)
        val_loss.append(total_val_loss)

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print(f" Denoise → Train: {avg_train_denoise:.6f}, Val: {val_loss_denoise:.6f}")
        print(f" Inpaint → Train: {avg_train_inpaint:.6f}, Val: {val_loss_inpaint:.6f}")
        print(f" Total → Train: {total_train_loss:.6f}, Val: {total_val_loss:.6f}")
        
        # --- Save best model ---
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(SAVE_ROOT, BEST_MODEL_FILE))
            print(f" Best model updated at epoch {epoch} (val loss = {best_val_loss:.6f})")

    # --- Plot losses ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_denoise, label="Train Denoise (MSE)")
    plt.plot(val_losses_denoise, label="Val Denoise (MSE)", linestyle="--")
    plt.plot(train_losses_inpaint, label="Train Inpaint (L1+Adv)")
    plt.plot(val_losses_inpaint, label="Val Inpaint (L1)", linestyle="--")
    plt.title("Training and Validation Losses (Separately)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, SEPARATE_LOSS_PLOT_FILE))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Total Train Loss")
    plt.plot(val_loss, label="Total Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, LOSS_PLOT_FILE))
    plt.close()


if __name__ == "__main__":
    main()
