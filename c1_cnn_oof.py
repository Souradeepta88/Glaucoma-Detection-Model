import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from fundus_dataset import FundusDataset, train_tf, eval_tf

torch.manual_seed(42)
np.random.seed(42)

base = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_csv(os.path.join(base, "image_index_224.csv"))
train_all = idx[idx["Set"] == "train"].reset_index(drop=True)
test_df = idx[idx["Set"] == "test"].reset_index(drop=True)

EPOCHS = 2

def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    return m

def predict(model, df):
    model.eval()
    loader = DataLoader(FundusDataset(df, eval_tf), batch_size=32)
    probs = []
    with torch.no_grad():
        for x, y in loader:
            probs += torch.softmax(model(x), dim=1)[:, 1].tolist()
    return np.array(probs)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_all))
test_preds = []

for fold, (tr_i, va_i) in enumerate(sgkf.split(train_all, train_all["Target"],
                                               groups=train_all["ID"]), start=1):
    tr_df, va_df = train_all.iloc[tr_i], train_all.iloc[va_i]
    counts = tr_df["Target"].value_counts().sort_index().values
    weights = torch.tensor(len(tr_df) / (2 * counts), dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(FundusDataset(tr_df, train_tf), batch_size=16, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(base, "models", f"resnet18_fold{fold}.pt"))
    oof[va_i] = predict(model, va_df)
    test_preds.append(predict(model, test_df))
    auc = roc_auc_score(va_df["Target"], oof[va_i])
    print("Fold %d | trained on %d eyes, predicted %d | fold AUC %.3f"
          % (fold, len(tr_df), len(va_df), auc))

train_all["CNN_prob"] = oof
print("\nOverall out-of-fold AUC on training set: %.3f"
      % roc_auc_score(train_all["Target"], oof))

test_df["CNN_prob"] = np.mean(test_preds, axis=0)

train_all[["ID", "Eye", "Target", "CNN_prob"]].to_csv(
    os.path.join(base, "cnn_oof_train.csv"), index=False)
test_df[["ID", "Eye", "Target", "CNN_prob"]].to_csv(
    os.path.join(base, "cnn_test.csv"), index=False)
print("Saved cnn_oof_train.csv and cnn_test.csv")