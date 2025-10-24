# ─────────────────────────────────────────────────────────────
# mini_imagenet_denoising_generator_splitwise.py
# Create Gaussian-noisy & clean image pairs for each split of Mini-ImageNet.
# ─────────────────────────────────────────────────────────────

import os, torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- Config ---
BASE_ROOT = "/mnt/DATA1/pankhi/DATA/mini_imagenet"   # parent dir containing train/validation/test
NOISE_STD = 0.1                                       # noise strength (0.05–0.2 recommended)

# --- Common Transform ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

# --- Noise function ---
def add_gaussian_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    noisy = torch.clamp(img + noise, 0., 1.)
    return noisy

# --- Process each split ---
for split in ["train", "validation", "test"]:
    src_root = os.path.join(BASE_ROOT, split)
    out_root = os.path.join(BASE_ROOT, f"{split}_denoising")
    os.makedirs(f"{out_root}/noisy", exist_ok=True)
    os.makedirs(f"{out_root}/clean", exist_ok=True)

    # load dataset
    dataset = datasets.ImageFolder(root=src_root, transform=transform)
    print(f"Processing {split}: {len(dataset)} images...")

    for i, (img, _) in enumerate(tqdm(dataset, desc=f"{split}")):
        noisy = add_gaussian_noise(img, NOISE_STD)
        TF.to_pil_image(noisy).save(f"{out_root}/noisy/{i:06d}.png")
        TF.to_pil_image(img).save(f"{out_root}/clean/{i:06d}.png")

    print(f"{split} denoising dataset created at: {out_root}")
