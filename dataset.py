import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import random

class EncoderInputDataset(Dataset):

    def __init__(self, root_dir, split="train", image_size=128, mask_ratio=0.25, num_patches=8):
        
        self.clean_dir = os.path.join(root_dir, "clean", split)
        self.noisy_dir = os.path.join(root_dir, "noisy", split)
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.num_patches = num_patches  # number of small rectangular patches
        self.classes = sorted(os.listdir(self.clean_dir))
        self.samples = []
        for cls in self.classes:
            clean_cls = os.path.join(self.clean_dir, cls)
            noisy_cls = os.path.join(self.noisy_dir, cls)
            for img_name in os.listdir(clean_cls):
                clean_path = os.path.join(clean_cls, img_name)
                noisy_path = os.path.join(noisy_cls, img_name)
                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    self.samples.append((clean_path, noisy_path))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    # def random_mask(self, img):
    #     mask = torch.ones_like(img)
    #     num_pixels = img.numel() // 3  # per-channel count
    #     mask_pixels = int(num_pixels * self.mask_ratio)
    #     idx = torch.randperm(num_pixels)[:mask_pixels]
    #     mask.view(-1)[idx] = 0
    #     masked_img = img * mask
    #     return masked_img, mask
    
    # def rectangular_mask(self, img):
    #     """Generate a rectangular mask instead of random pixel mask."""
    #     _, H, W = img.shape
    #     mask = torch.ones_like(img)

    #     # Compute rectangular dimensions based on mask_ratio (area)
    #     mask_area = int(H * W * self.mask_ratio)
    #     rect_w = int((mask_area) ** 0.5)
    #     rect_h = rect_w  # make it square (you can randomize this for more variety)

    #     # Random top-left corner of the rectangle
    #     x1 = random.randint(0, W - rect_w)
    #     y1 = random.randint(0, H - rect_h)

    #     # Mask region
    #     mask[:, y1:y1 + rect_h, x1:x1 + rect_w] = 0

    #     masked_img = img * mask
    #     return masked_img, mask

    def multi_rectangular_mask(self, img):
        """Generate multiple small rectangular masks over the image."""
        _, H, W = img.shape
        mask = torch.ones_like(img)

        total_mask_area = int(H * W * self.mask_ratio)
        patch_area = total_mask_area // self.num_patches

        for _ in range(self.num_patches):
            # Randomize aspect ratio for variation
            aspect_ratio = random.uniform(0.5, 2.0)
            rect_w = int((patch_area * aspect_ratio) ** 0.5)
            rect_h = int(patch_area / (rect_w + 1e-6))

            # Avoid going out of bounds
            rect_w = min(rect_w, W - 1)
            rect_h = min(rect_h, H - 1)

            # Random position
            x1 = random.randint(0, max(0, W - rect_w - 1))
            y1 = random.randint(0, max(0, H - rect_h - 1))

            # Apply mask
            mask[:, y1:y1 + rect_h, x1:x1 + rect_w] = 0

        masked_img = img * mask
        return masked_img, mask

    def __getitem__(self, idx):
        clean_path, noisy_path = self.samples[idx]

        clean = self.transform(Image.open(clean_path).convert("RGB"))
        noisy = self.transform(Image.open(noisy_path).convert("RGB"))
        masked_img, mask = self.multi_rectangular_mask(clean)

        # Combine all three as encoder input: [3(clean) + 3(noisy) + 3(masked_clean)] = [9, H, W]
        combined_input = torch.cat([clean, noisy, masked_img], dim=0)

        return {
            "input": combined_input,   # [9, H, W]
            "clean": clean,            # [3, H, W] target
            "noisy": noisy,            # [3, H, W] noisy input (for visualization or loss)
            "mask": masked_img               # [3, H, W] binary mask 
        }


def get_dataloader(root_dir, split="train", batch_size=16, image_size=128, mask_ratio=0.25, num_workers=4, num_patches=8):
    dataset = EncoderInputDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        mask_ratio=mask_ratio,
        num_patches=num_patches
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# if __name__ == "__main__":
#     root = "/mnt/DATA1/vivekananda/DATA/mini_imagenet_denoising"
#     train_loader = get_dataloader(root, split="train", batch_size=16)
#     val_loader = get_dataloader(root, split="validation", batch_size=16)
#     for batch in train_loader:
#         print("Input:", batch["input"].shape)  # [B, 9, H, W]
#         print("Clean:", batch["clean"].shape)  # [B, 3, H, W]
#         print("Noisy:", batch["noisy"].shape)  # [B, 3, H, W]
#         print("Mask:", batch["mask"].shape)    # [B, 3, H, W]
#         break
