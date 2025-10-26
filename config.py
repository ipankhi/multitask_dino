import os

# Root directory for data (clean, noisy, mask images)
DATA_ROOT = "/mnt/DATA1/vivekananda/DATA/mini_imagenet_denoising"

# Base project directory
PROJECT_ROOT = "/mnt/DATA1/vivekananda/multitask_dino"

# Directory to save checkpoints
SAVE_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(SAVE_ROOT, exist_ok=True)

# Directory to save test visualizations and results
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_results_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional directory to save logs or TensorBoard runs
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CKPT_PATH = os.path.join(SAVE_ROOT, "best_model_2.pth")

# ─────────────────────────────────────────────
#   MODEL CONFIGURATION
# ─────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.dino"  # DINO ViT backbone from timm
IMAGE_SIZE = 128  # Input image size (H, W)
PATCH_SIZE = 16   # ViT patch size
EMBED_DIM = 768   # Embedding dimension (depends on ViT variant)
FREEZE_BACKBONE = True  # Freeze pretrained ViT weights

# ─────────────────────────────────────────────
#   LORA CONFIGURATION
# ─────────────────────────────────────────────
# LoRA adapter parameters applied to the last ViT block
LORA_R = 8          # Rank (low-rank dimension)
LORA_ALPHA = 16     # Scaling factor

# ─────────────────────────────────────────────
#  DATASET / DATALOADER CONFIGURATION
# ─────────────────────────────────────────────
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 64
BATCH_SIZE_TEST = 64  # for faster evaluation
NUM_WORKERS = 4       # parallel data loading threads
PIN_MEMORY = True


# ─────────────────────────────────────────────
#  TRAINING CONFIGURATION
# ─────────────────────────────────────────────
EPOCHS = 20
LEARNING_RATE = 1e-4
LR_DISCRIMINATOR = LEARNING_RATE * 0.5  # smaller LR for discriminator


# Loss weight scaling factors
LAMBDA_ADV = 0.1  # weight for adversarial loss (inpainting)
LAMBDA_L1 = 0.9     # weight for L1 loss (inpainting)


# ─────────────────────────────────────────────
#   LOGGING / CHECKPOINT CONFIGURATION
# ─────────────────────────────────────────────

BEST_MODEL_FILE = "best_model_2.pth"
LOSS_PLOT_FILE = "loss_plot_2.png"
SEPARATE_LOSS_PLOT_FILE = "separate_loss_plot_2.png"
METRICS_PATH_FILE = "test_metrics_2.txt"

# ─────────────────────────────────────────────
#   TEST CONFIGURATION
# ─────────────────────────────────────────────
NUM_VISUAL_SAMPLES = 10  # Number of test samples to visualize
MASK_RATIO = 0.25       # Total masked area
NUM_PATCHES = 8         # No of small patches of mask

# ─────────────────────────────────────────────
#   DEVICE CONFIGURATION
# ─────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

