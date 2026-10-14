import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fundus_dataset import FundusDataset, train_tf, eval_tf

torch.manual_seed(42)
np.random.seed(42)

base = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_csv(os.path.join(base, "image_index_224.csv"))
train_all = idx[idx["Set"] == "train"].reset_index(drop=True)

# hold out ~20% of the TRAINING patients for validation (never the test set)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_i, va_i = next(sgkf.split(train_all, train_all["Target"], groups=train_all["ID"]))
tr_df, va_df = train_all.iloc[tr_i], train_all.iloc[va_i]

print("Patients in both train and validation (must be 0):",
      len(set(tr_df["ID"]) & set(va_df["ID"])))
print("Train:", len(tr_df), "rows,", int(tr_df["Target"].sum()), "glaucoma")
print("Val:  ", len(va_df), "rows,", int(va_df["Target"].sum()), "glaucoma\n")

train_loader = DataLoader(FundusDataset(tr_df, train_tf), batch_size=16, shuffle=True)
val_loader = DataLoader(FundusDataset(va_df, eval_tf), batch_size=32)

# pretrained ResNet18 with a new 2-class output layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))

# weighted loss: the rarer class (glaucoma) counts more
counts = tr_df["Target"].value_counts().sort_index().values
weights = torch.tensor(len(tr_df) / (2 * counts), dtype=torch.float)
print("Class weights (healthy, glaucoma):", weights.numpy().round(2), "\n")
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def run_eval(loader):
    model.eval()
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            losses.append(criterion(out, y).item() * len(y))
            probs += torch.softmax(out, dim=1)[:, 1].tolist()
            labels += y.tolist()
    return sum(losses) / len(labels), np.array(probs), np.array(labels)

EPOCHS, PATIENCE = 15, 4
best_auc, bad, history = 0, 0, []
os.makedirs(os.path.join(base, "models"), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    train_loss = total / len(tr_df)

    val_loss, probs, labels = run_eval(val_loader)
    auc = roc_auc_score(labels, probs)
    rec = recall_score(labels, probs > 0.5)
    history.append((train_loss, val_loss, auc))
    print("Epoch %2d | train loss %.3f | val loss %.3f | val AUC %.3f | val recall %.2f | %.0fs"
          % (epoch, train_loss, val_loss, auc, rec, time.time() - t0))

    if auc > best_auc:
        best_auc, bad = auc, 0
        torch.save(model.state_dict(), os.path.join(base, "models", "resnet18_best.pt"))
    else:
        bad += 1
        if bad >= PATIENCE:
            print("No improvement for", PATIENCE, "epochs, stopping early.")
            break

print("\nBest validation AUC: %.3f (weights saved to models/resnet18_best.pt)" % best_auc)

# learning curves
h = np.array(history)
os.makedirs(os.path.join(base, "results"), exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h[:, 0], label="Train loss")
ax[0].plot(h[:, 1], label="Validation loss")
ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].set_title("Loss")
ax[1].plot(h[:, 2], color="green")
ax[1].set_xlabel("Epoch"); ax[1].set_title("Validation ROC-AUC")
plt.tight_layout()
plt.savefig(os.path.join(base, "results", "cnn_training_curves.png"), dpi=150)
print("Saved results/cnn_training_curves.png")