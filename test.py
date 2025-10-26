# test.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from math import log10

from model import DenoiseInpaintModel
from dataset import get_dataloader
from config import *

# ─────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────
def mse_loss(pred, target):
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2).item()

def mae_loss(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def psnr(pred, target):
    """Peak Signal-to-Noise Ratio"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1 / mse.item())

def ssim_metric(pred, target):
    """Structural Similarity Index (SSIM)"""
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()
    return ssim(pred_np, target_np, channel_axis=2, data_range=1.0)


# ─────────────────────────────────────────────
# Visualization Function
# ─────────────────────────────────────────────


def visualize_results(clean, noisy, mask, denoised, inpainted, save_dir, idx):
    """
    Save side-by-side comparison of:
    Clean | Noisy | Mask | Masked Clean | Denoised | Inpainted
    """

    # Convert tensors to numpy images for plotting
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    titles = ["Clean", "Noisy", "Masked", "Denoised", "Inpainted"]
    imgs = [clean, noisy, mask, denoised, inpainted]

    for ax, img, title in zip(axes, imgs, titles):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 1:  # for mask
            img_np = np.repeat(img_np, 3, axis=-1)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Testing Routine
# ─────────────────────────────────────────────
def main():
    

    # --- Load Best Model ---
    print("\n  Loading best model for testing...")
    model = DenoiseInpaintModel(
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        pretrained=True
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    print("  Model successfully loaded from checkpoint!")

    # --- Load Test Dataset ---
    test_loader = get_dataloader(
        DATA_ROOT, split="test", batch_size=BATCH_SIZE_TEST, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO, num_patches=NUM_PATCHES
    )

    # --- Metric Accumulators ---
    metrics = {
        "denoise": {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0},
        "inpaint": {"mse": 0, "mae": 0, "psnr": 0, "ssim": 0}
    }
    count = 0
    visualized = 0  # Counter for visualization samples

    # --- Testing Loop ---
    print("\n  Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Progress")):
            noisy = batch["noisy"].to(DEVICE)
            clean = batch["clean"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            outputs = model(noisy, mask)
            denoised, inpainted = outputs["denoised"], outputs["inpainted"]

            batch_size = noisy.size(0)
            count += batch_size

            # --- Compute metrics for the entire batch ---
            for i in range(batch_size):
                metrics["denoise"]["mse"] += mse_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["mae"] += mae_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["psnr"] += psnr(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["ssim"] += ssim_metric(denoised[i], clean[i])

                metrics["inpaint"]["mse"] += mse_loss(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["mae"] += mae_loss(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["psnr"] += psnr(inpainted[i:i+1], clean[i:i+1])
                metrics["inpaint"]["ssim"] += ssim_metric(inpainted[i], clean[i])

            # --- Save visualizations for 10 total samples (not per batch) ---
            if visualized < NUM_VISUAL_SAMPLES:
                num_to_save = min(NUM_VISUAL_SAMPLES - visualized, batch_size)
                for j in range(num_to_save):
                    visualize_results(
                        clean[j], noisy[j], mask[j], denoised[j], inpainted[j],
                        save_dir=OUTPUT_DIR, idx=visualized
                    )
                    visualized += 1

            # Stop early once we have saved 10 visualizations
            if visualized >= NUM_VISUAL_SAMPLES:
                continue

    # --- Compute Average Metrics ---
    for task in ["denoise", "inpaint"]:
        for key in metrics[task]:
            metrics[task][key] /= count

    # --- Print Summary ---
    print("\n  ===== TEST PERFORMANCE SUMMARY =====")
    print(f"   DENOISING METRICS:")
    print(f"   • MSE  : {metrics['denoise']['mse']:.6f}")
    print(f"   • MAE  : {metrics['denoise']['mae']:.6f}")
    print(f"   • PSNR : {metrics['denoise']['psnr']:.2f} dB")
    print(f"   • SSIM : {metrics['denoise']['ssim']:.4f}")
    print("——————————————————————————————————————")
    print(f"   INPAINTING METRICS:")
    print(f"   • MSE  : {metrics['inpaint']['mse']:.6f}")
    print(f"   • MAE  : {metrics['inpaint']['mae']:.6f}")
    print(f"   • PSNR : {metrics['inpaint']['psnr']:.2f} dB")
    print(f"   • SSIM : {metrics['inpaint']['ssim']:.4f}")
    print("——————————————————————————————————————")
    print(f"  Visualization saved for 10 samples → {OUTPUT_DIR}")

    # --- Save Metrics to File ---
    metrics_path = os.path.join(OUTPUT_DIR, METRICS_PATH_FILE)
    with open(metrics_path, "w") as f:
        f.write("===== TEST PERFORMANCE SUMMARY =====\n\n")
        f.write("[Denoising]\n")
        for k, v in metrics["denoise"].items():
            f.write(f"{k.upper()}: {v:.6f}\n")
        f.write("\n[Inpainting]\n")
        for k, v in metrics["inpaint"].items():
            f.write(f"{k.upper()}: {v:.6f}\n")
    print(f"  Metrics saved to: {metrics_path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
