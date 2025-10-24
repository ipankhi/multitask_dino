# ─────────────────────────────────────────────────────────────
# mini_imagenet_denoising_generator.py
# Create synthetic Gaussian-noisy versions of Mini-ImageNet images.
# ─────────────────────────────────────────────────────────────

import os, torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# --- Config ---
SRC_ROOT = "path/to/mini_imagenet/train"      # change to your Mini-ImageNet root
OUT_ROOT = "mini_imagenet_denoising"          # output directory
NOISE_STD = 0.1                               # noise strength (0.05–0.2 recommended)

os.makedirs(f"{OUT_ROOT}/noisy", exist_ok=True)
os.makedirs(f"{OUT_ROOT}/clean", exist_ok=True)

# --- Dataset loader ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=SRC_ROOT, transform=transform)

# --- Noise function ---
def add_gaussian_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    noisy = torch.clamp(img + noise, 0., 1.)
    return noisy

# --- Generate noisy/clean pairs ---
for i, (img, _) in enumerate(dataset):
    noisy = add_gaussian_noise(img, NOISE_STD)
    TF.to_pil_image(noisy).save(f"{OUT_ROOT}/noisy/{i:06d}.png")
    TF.to_pil_image(img).save(f"{OUT_ROOT}/clean/{i:06d}.png")

print(f"Denoising dataset created at: {OUT_ROOT}")
