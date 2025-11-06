# ─────────────────────────────────────────────
# test.py — Joint Denoising + Inpainting Evaluation (Fixed & Aligned)
# Evaluates MSE, MAE, PSNR, SSIM and saves visual comparisons
# ─────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from math import log10
import torch.nn.functional as F

from model import DenoiseInpaintModel
from dataloader import get_dataloader
from config import *

# ─────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────
def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2).item()

def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())

def ssim_metric(pred, target):
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()
    return ssim(pred_np, target_np, channel_axis=2, data_range=1.0)

# ─────────────────────────────────────────────
# Visualization Function
# ─────────────────────────────────────────────
def visualize_results(clean, noisy, masked, mask, denoised, inpainted, save_dir, idx):
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    titles = ["Clean", "Noisy", "Masked", "Mask", "Denoised", "Inpainted"]
    imgs = [clean, noisy, masked, mask, denoised, inpainted]

    for ax, img, title in zip(axes, imgs, titles):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────
# Testing Routine
# ─────────────────────────────────────────────
def main():
    print("\n🔹 Loading model and fusion conv for testing...")
    model = DenoiseInpaintModel(
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        pretrained=True
    ).to(DEVICE)

    # Load trained fusion_conv
    fusion_conv = nn.Conv2d(7, 3, kernel_size=1).to(DEVICE)

    print(f"🔹 Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_state = model.state_dict()

    # Extract fusion conv weights if saved jointly
    if "fusion_conv.weight" in ckpt:
        fusion_conv.load_state_dict({
            "weight": ckpt["fusion_conv.weight"],
            "bias": ckpt["fusion_conv.bias"]
        })
        del ckpt["fusion_conv.weight"], ckpt["fusion_conv.bias"]

    # Skip mismatched keys safely
    for k in list(ckpt.keys()):
        if k in model_state and ckpt[k].shape != model_state[k].shape:
            print(f"⚠️ Skipping mismatched key: {k} ({ckpt[k].shape} → {model_state[k].shape})")
            ckpt.pop(k)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"✅ Model loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)\n")

    model.eval(); fusion_conv.eval()

    test_loader = get_dataloader(
        DATA_ROOT, split="test",
        batch_size=BATCH_SIZE_TEST,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = {
        "denoise": {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0},
        "inpaint": {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0}
    }
    count, visualized = 0, 0

    print("🔹 Running inference on test set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Progress")):
            noisy = batch["noisy"].to(DEVICE)
            masked = batch["masked"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)

            # --- Fuse degraded inputs using trained conv ---
            x_fused = torch.cat([noisy, masked, mask], dim=1)
            x_fused = fusion_conv(x_fused)

            outputs = model(x_fused, mask=mask)
            denoised, inpainted = outputs["denoised"], outputs["inpainted"]

            # Resize outputs (ensure 128×128)
            denoised = F.interpolate(denoised, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            inpainted = F.interpolate(inpainted, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

            batch_size = noisy.size(0)
            count += batch_size

            # --- Metrics ---
            for i in range(batch_size):
                metrics["denoise"]["mse"] += mse_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["mae"] += mae_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["psnr"] += psnr(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["ssim"] += ssim_metric(denoised[i], clean[i])

                metrics["inpaint"]["mse"] += mse_loss(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["mae"] += mae_loss(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["psnr"] += psnr(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["ssim"] += ssim_metric(inpainted[i], clean[i])

                if visualized < NUM_VISUAL_SAMPLES:
                    visualize_results(
                        clean[i], noisy[i], masked[i], mask[i],
                        denoised[i], inpainted[i],
                        save_dir=OUTPUT_DIR, idx=visualized
                    )
                    visualized += 1

    # Average metrics
    for task in ["denoise", "inpaint"]:
        for key in metrics[task]:
            metrics[task][key] /= count

    # Print summary
    print("\n📊 ===== TEST PERFORMANCE SUMMARY =====")
    print(f"🧩 DENOISING:")
    print(f"  MSE  : {metrics['denoise']['mse']:.6f}")
    print(f"  MAE  : {metrics['denoise']['mae']:.6f}")
    print(f"  PSNR : {metrics['denoise']['psnr']:.2f} dB")
    print(f"  SSIM : {metrics['denoise']['ssim']:.4f}")
    print("——————————————————————————————————————")
    print(f"🎨 INPAINTING:")
    print(f"  MSE  : {metrics['inpaint']['mse']:.6f}")
    print(f"  MAE  : {metrics['inpaint']['mae']:.6f}")
    print(f"  PSNR : {metrics['inpaint']['psnr']:.2f} dB")
    print(f"  SSIM : {metrics['inpaint']['ssim']:.4f}")
    print("——————————————————————————————————————")
    print(f"📸 Visualizations saved → {OUTPUT_DIR}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, METRICS_PATH_FILE)
    with open(metrics_path, "w") as f:
        f.write("===== TEST PERFORMANCE SUMMARY =====\n\n")
        f.write("[Denoising]\n")
        for k, v in metrics["denoise"].items():
            f.write(f"{k.upper()}: {v:.6f}\n")
        f.write("\n[Inpainting]\n")
        for k, v in metrics["inpaint"].items():
            f.write(f"{k.upper()}: {v:.6f}\n")
    print(f"📝 Metrics saved to: {metrics_path}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()