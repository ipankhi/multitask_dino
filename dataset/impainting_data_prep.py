# ─────────────────────────────────────────────────────────────
# mini_imagenet_inpainting_generator_hierarchical.py
# Create masked-image / mask / clean triplets for Mini-ImageNet,
# preserving split & class folder hierarchy.
# ─────────────────────────────────────────────────────────────

import os, numpy as np
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- Config ---
BASE_ROOT = "/mnt/DATA1/pankhi/DATA/mini_imagenet"         # contains train/validation/test
OUT_ROOT  = "/mnt/DATA1/pankhi/DATA/mini_imagenet_inpainting"
MASK_RATIO = 0.3                                            # fraction of image to mask
IMG_SIZE = (128, 128)                                       # resize target

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# --- Mask generator (rectangular) ---
def random_mask(size, ratio=0.3):
    w, h = size
    mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(mask)
    num_rects = np.random.randint(1, 4)
    for _ in range(num_rects):
        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
        x2, y2 = np.random.randint(w//2, w), np.random.randint(h//2, h)
        draw.rectangle([x1, y1, x2, y2], fill=0)
    return mask

# --- Prepare main folders ---
for mode in ["masked", "mask", "clean"]:
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, mode, split), exist_ok=True)

# --- Process each split ---
for split in ["train", "validation", "test"]:
    src_root = os.path.join(BASE_ROOT, split)
    dataset = datasets.ImageFolder(root=src_root, transform=transform)
    classes = dataset.classes

    print(f"\nProcessing {split}: {len(dataset)} images across {len(classes)} classes...")

    # Create subfolders for each class
    for cls in classes:
        for mode in ["masked", "mask", "clean"]:
            os.makedirs(os.path.join(OUT_ROOT, mode, split, cls), exist_ok=True)

    # Generate masked / mask / clean triplets
    for i, (img_tensor, label) in enumerate(tqdm(dataset, desc=f"{split}")):
        cls = classes[label]
        img = TF.to_pil_image(img_tensor)
        mask = random_mask(img.size, MASK_RATIO)
        masked = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask)

        masked.save(os.path.join(OUT_ROOT, "masked", split, cls, f"{i:06d}.png"))
        mask.save(os.path.join(OUT_ROOT, "mask", split, cls, f"{i:06d}.png"))
        img.save(os.path.join(OUT_ROOT, "clean", split, cls, f"{i:06d}.png"))

    print(f"✅ Completed {split}")

print(f"\nAll inpainting datasets created successfully under: {OUT_ROOT}")
