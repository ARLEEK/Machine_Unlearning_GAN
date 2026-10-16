from custom_gan import train_gan
from data_loader_coco import load_coco_images
import torch
import os
import random

# hyperparameters
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 25
NZ = 100               # latent vector size
LR = 2e-4
BETA1 = 0.5
SAVE_EVERY = 5
SAMPLE_SIZE = 64
IMAGE_SIZE = 64
DATA_ROOT = "./data/coco"
OUT_DIR = "./runs/dcgan"
NUM_WORKERS = 4

# setting values
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#   Load COCO dataset
dataloader = load_coco_images(
    root=DATA_ROOT,
    image_set="train2017",
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    image_size=IMAGE_SIZE
)

# create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# training custom gan
train_gan(
    dataloader=dataloader,
    device=device,
    nz=NZ,
    n_epochs=NUM_EPOCHS,
    lr=LR,
    beta1=BETA1,
    out_dir=OUT_DIR,
    save_every=SAVE_EVERY,
    sample_size=SAMPLE_SIZE
)