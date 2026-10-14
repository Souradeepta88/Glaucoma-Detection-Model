import pandas as pd
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
img_root = os.path.join(base, "PapilaDB", "FundusImages")

# 1. find every image file
files = []
for root, dirs, fs in os.walk(img_root):
    for f in fs:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(root, f))
print("Image files found:", len(files))
print("First 5 names:", [os.path.basename(f) for f in sorted(files)[:5]])
print()

# 2. read patient number and eye from the filename (e.g. RET002OD.jpg)
rows = []
for f in files:
    m = re.search(r"(\d+)\s*(OD|OS)", os.path.basename(f), re.IGNORECASE)
    if m:
        rows.append({"num": int(m.group(1)), "Eye": 0 if m.group(2).upper() == "OD" else 1, "path": f})
imgs = pd.DataFrame(rows)
print("Filenames parsed:", len(imgs), "of", len(files))

# 3. match to our modelling table
data = pd.read_csv(os.path.join(base, "model_data.csv"))
data["num"] = data["ID"].str.replace("#", "").astype(int)
split = pd.read_csv(os.path.join(base, "split_ids.csv"))

merged = data[["ID", "num", "Eye", "Target"]].merge(imgs, on=["num", "Eye"], how="left")
merged = merged.merge(split, on="ID", how="left")

print("Table rows:", len(merged))
print("Rows with an image:", merged["path"].notna().sum())
print("Rows missing an image:", merged["path"].isna().sum())
print()
print(merged.groupby(["Set", "Target"]).size())

# 4. image sizes (check 15 random ones)
sizes = [Image.open(p).size for p in merged["path"].dropna().sample(15, random_state=1)]
print("\nSample image sizes (width, height):", sorted(set(sizes)))

# 5. save the index for the next steps
merged.to_csv(os.path.join(base, "image_index.csv"), index=False)

# 6. picture of 2 healthy + 2 glaucoma eyes
os.makedirs(os.path.join(base, "results"), exist_ok=True)
sample = pd.concat([merged[merged["Target"] == 0].dropna(subset=["path"]).head(2),
                    merged[merged["Target"] == 1].dropna(subset=["path"]).head(2)])
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (_, r) in zip(axes, sample.iterrows()):
    im = Image.open(r["path"]).convert("RGB")
    im.thumbnail((500, 500))
    ax.imshow(im)
    ax.set_title(r["ID"] + (" OD" if r["Eye"] == 0 else " OS") + " - " + ("Glaucoma" if r["Target"] == 1 else "Healthy"))
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(base, "results", "sample_images.png"), dpi=120)
print("\nSaved image_index.csv and results/sample_images.png")

import torch, torchvision
print("\nPyTorch", torch.__version__, "| torchvision", torchvision.__version__, "| GPU available:", torch.cuda.is_available())