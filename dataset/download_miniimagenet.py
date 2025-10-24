from datasets import load_dataset
import os

dataset_path = "/mnt/DATA1/pankhi/DATA/mini_imagenet"
os.mkdir(dataset_path)
ds = load_dataset("timm/mini-imagenet", cache_dir=dataset_path)
print(ds)
