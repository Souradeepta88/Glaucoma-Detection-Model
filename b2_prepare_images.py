import pandas as pd
import numpy as np
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from fundus_dataset import FundusDataset, train_tf, eval_tf, MEAN, STD

base = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_csv(os.path.join(base, "image_index.csv"))

# 1. resize every image once and save it
out_dir = os.path.join(base, "images_224")
os.makedirs(out_dir, exist_ok=True)

small_paths = []
for i, r in idx.iterrows():
    name = r["ID"].replace("#", "") + ("_OD" if r["Eye"] == 0 else "_OS") + ".jpg"
    out = os.path.join(out_dir, name)
    if not os.path.exists(out):
        Image.open(r["path"]).convert("RGB").resize((224, 224), Image.LANCZOS).save(out, quality=95)
    small_paths.append(out)
    if (i + 1) % 100 == 0:
        print("Resized", i + 1, "of", len(idx))

idx["small_path"] = small_paths
idx.to_csv(os.path.join(base, "image_index_224.csv"), index=False)

# 2. check the datasets and loaders work
train_df = idx[idx["Set"] == "train"]
test_df = idx[idx["Set"] == "test"]
train_ds = FundusDataset(train_df, train_tf)
test_ds = FundusDataset(test_df, eval_tf)
print("\nTrain images:", len(train_ds), "| Test images:", len(test_ds))

x, y = next(iter(DataLoader(train_ds, batch_size=16, shuffle=True)))
print("One batch ->", tuple(x.shape), "labels:", y.tolist())
print("Pixel mean per channel (should be near 0):", x.mean(dim=(0, 2, 3)).numpy().round(2))

# 3. picture: one original and 5 augmented versions of the same eye
orig = Image.open(train_df.iloc[0]["small_path"]).convert("RGB")
fig, axes = plt.subplots(1, 6, figsize=(18, 3.3))
axes[0].imshow(orig)
axes[0].set_title("Original")
for k in range(1, 6):
    t = train_tf(orig)
    img = t * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)   # undo normalisation to display
    axes[k].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[k].set_title("Augmented " + str(k))
for a in axes:
    a.axis("off")
plt.tight_layout()
os.makedirs(os.path.join(base, "results"), exist_ok=True)
plt.savefig(os.path.join(base, "results", "augmentations.png"), dpi=120)

print("\nSaved image_index_224.csv, images_224/ and results/augmentations.png")