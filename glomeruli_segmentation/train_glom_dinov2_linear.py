import os
import time
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import openslide
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from torchvision import transforms

# -----------------------
# Paths (EDIT THESE)
# -----------------------
NDPI_PATH = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/data/lab_kidney_data_ndpi/OTS-24-22043 - 2024-08-28 15.08.37.ndpi"
BOXES_CSV = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/final_glomeruli_segmentation_pipeline/out/region_split_max3.csv"
LABELS_CSV = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/final_glomeruli_segmentation_pipeline/out/box_labels.csv"
OUTDIR = "/Users/edmundtsou/Desktop/JEFworks/jefworks-structure_segmentation/final_glomeruli_segmentation_pipeline/out"
os.makedirs(OUTDIR, exist_ok=True)

CKPT_PATH = os.path.join(OUTDIR, "glom_dinov2_linear.pt")

# -----------------------
# Config
# -----------------------
SEED = 7
BATCH = 32
EPOCHS = 80
LR = 2e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Mac: keep small unless you know OpenSlide is happy

# Crop behavior
PAD_FRAC = 0.05         # a bit of context around bbox
TARGET_VIEW_PX = 900    # crop read downsample target
FINAL_IMG_SIZE = 224    # DINOv2 input

# Label handling
KEEP_LABELS = {"glomerulus": 1, "not_glomerulus": 0}
DROP_UNSURE = True
KEEP_MULTI_AS_POSITIVE = True  # if False, drops multi=True samples

# Train/val split
VAL_FRAC = 0.2

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# -----------------------
# Utilities
# -----------------------
def expand_box(x0, y0, x1, y1, pad_frac):
    w = x1 - x0
    h = y1 - y0
    px = int(round(w * pad_frac))
    py = int(round(h * pad_frac))
    return x0 - px, y0 - py, x1 + px, y1 + py

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, min(W, int(round(x0))))
    y0 = max(0, min(H, int(round(y0))))
    x1 = max(0, min(W, int(round(x1))))
    y1 = max(0, min(H, int(round(y1))))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def read_crop(slide: openslide.OpenSlide, x0, y0, x1, y1, target_view_px=900):
    """Read a bbox crop using best pyramid level so largest dim ~ target_view_px."""
    W0, H0 = slide.dimensions
    x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W0, H0)
    w = x1 - x0
    h = y1 - y0

    want_down = max(w, h) / float(target_view_px)
    level = slide.get_best_level_for_downsample(want_down)
    down = float(slide.level_downsamples[level])

    x0_l = int(x0 / down)
    y0_l = int(y0 / down)
    w_l  = max(1, int(w / down))
    h_l  = max(1, int(h / down))

    img = slide.read_region((x0_l, y0_l), level, (w_l, h_l)).convert("RGB")
    return img

# -----------------------
# Dataset
# -----------------------
class GlomBoxDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, ndpi_path: str, transform, pad_frac=0.05, target_view_px=900):
        self.df = manifest.reset_index(drop=True)
        self.ndpi_path = ndpi_path
        self.transform = transform
        self.pad_frac = pad_frac
        self.target_view_px = target_view_px
        self._slide = None

    def _get_slide(self):
        # OpenSlide objects are not picklable; open per-worker lazily
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.ndpi_path)
        return self._slide

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        x0, y0, x1, y1 = expand_box(x0, y0, x1, y1, self.pad_frac)

        slide = self._get_slide()
        img = read_crop(slide, x0, y0, x1, y1, target_view_px=self.target_view_px)

        x = self.transform(img)
        y = int(r.y)
        return x, y

# -----------------------
# Build manifest from your files
# -----------------------
def main():
    boxes = pd.read_csv(BOXES_CSV)
    labs = pd.read_csv(LABELS_CSV)

    if "id" not in boxes.columns:
        raise ValueError("Boxes CSV must have an 'id' column.")
    if "id" not in labs.columns:
        raise ValueError("Labels CSV must have an 'id' column.")
    boxes["id"] = boxes["id"].astype(int)
    labs["id"] = labs["id"].astype(int)

    # keep confident labels
    labs = labs[labs["label"].isin(list(KEEP_LABELS.keys()))].copy()
    labs["y"] = labs["label"].map(KEEP_LABELS).astype(int)

    if "multi" in labs.columns and (not KEEP_MULTI_AS_POSITIVE):
        labs = labs[~labs["multi"].astype(bool)].copy()

    m = boxes.merge(labs[["id", "y"]], on="id", how="inner").copy()
    if len(m) == 0:
        raise RuntimeError("No matched rows between boxes and labels. Check your CSV paths and ids.")

    print("Total labeled samples:", len(m))
    print(m["y"].value_counts().rename({0:"not_glomerulus", 1:"glomerulus"}))

    # stratified split
    train_df, val_df = train_test_split(
        m, test_size=VAL_FRAC, random_state=SEED, stratify=m["y"]
    )

    # -----------------------
    # Transforms
    # -----------------------
    # DINOv2 expects ImageNet-style normalization
    transform_train = transforms.Compose([
        transforms.Resize(FINAL_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(FINAL_IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(FINAL_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(FINAL_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    train_ds = GlomBoxDataset(train_df, NDPI_PATH, transform_train, pad_frac=PAD_FRAC, target_view_px=TARGET_VIEW_PX)
    val_ds   = GlomBoxDataset(val_df, NDPI_PATH, transform_val,   pad_frac=PAD_FRAC, target_view_px=TARGET_VIEW_PX)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # -----------------------
    # Model: Frozen DINOv2 + linear head
    # -----------------------
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load DINOv2 backbone from torch hub
    # You can change to dinov2_vitb14 for stronger but heavier.
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    # Infer embedding dim with a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, 3, FINAL_IMG_SIZE, FINAL_IMG_SIZE).to(device)
        emb = backbone(dummy)
        emb_dim = emb.shape[1]

    head = nn.Linear(emb_dim, 1).to(device)

    # -----------------------
    # Training loop
    # -----------------------
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_eval():
        head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).float()
                z = backbone(x)
                logits = head(z).squeeze(1)
                prob = torch.sigmoid(logits)
                ys.append(y.cpu().numpy())
                ps.append(prob.cpu().numpy())
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        # AUROC requires both classes in val
        auc = roc_auc_score(y_true, y_prob) if (len(np.unique(y_true)) == 2) else float("nan")
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        return auc, acc

    best_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            with torch.no_grad():
                z = backbone(x)

            logits = head(z).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        auc, acc = run_eval()
        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | val_auc={auc:.4f} | val_acc={acc:.4f}")

        # Save best by AUC (fallback to acc if AUC nan)
        score = auc if not math.isnan(auc) else acc
        if score > best_auc:
            best_auc = score
            torch.save(
                {
                    "backbone": "dinov2_vits14",
                    "head_state": head.state_dict(),
                    "emb_dim": emb_dim,
                    "pad_frac": PAD_FRAC,
                    "target_view_px": TARGET_VIEW_PX,
                    "final_img_size": FINAL_IMG_SIZE,
                    "threshold": 0.5,
                    "val_auc": float(auc) if not math.isnan(auc) else None,
                    "val_acc": float(acc),
                    "timestamp": int(time.time()),
                },
                CKPT_PATH
            )
            print("  saved:", CKPT_PATH)

    print("Done. Best checkpoint:", CKPT_PATH)

if __name__ == "__main__":
    main()