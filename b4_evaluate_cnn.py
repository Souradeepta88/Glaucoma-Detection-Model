import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fundus_dataset import FundusDataset, eval_tf

base = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_csv(os.path.join(base, "image_index_224.csv"))
test_img = idx[idx["Set"] == "test"].reset_index(drop=True)

# 1. load the trained CNN
model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
model.load_state_dict(torch.load(os.path.join(base, "models", "resnet18_best.pt"),
                                 map_location="cpu"))
model.eval()

# 2. CNN probabilities on the test images
loader = DataLoader(FundusDataset(test_img, eval_tf), batch_size=32)
probs = []
with torch.no_grad():
    for x, y in loader:
        probs += torch.softmax(model(x), dim=1)[:, 1].tolist()
test_img["CNN"] = probs

# 3. tabular model probabilities on the same eyes
test = pd.read_csv(os.path.join(base, "test.csv"))
X_test = test.drop(columns=["ID", "Target"])
for name, f in {"SVM": "svm.joblib", "Random Forest": "random_forest.joblib",
                "XGBoost": "xgboost.joblib"}.items():
    pipe = joblib.load(os.path.join(base, "models", f))
    test[name] = pipe.predict_proba(X_test)[:, 1]

# 4. line everything up by patient ID and eye
df = test_img[["ID", "Eye", "Target", "CNN"]].merge(
    test[["ID", "Eye", "SVM", "Random Forest", "XGBoost"]], on=["ID", "Eye"])
print("Test eyes:", len(df), "| glaucoma:", int(df["Target"].sum()), "\n")
y_true = df["Target"].values

# 5. metrics for every model
rows = []
os.makedirs(os.path.join(base, "results"), exist_ok=True)
plt.figure(1, figsize=(6, 5))
for name in ["CNN (ResNet18)", "SVM", "Random Forest", "XGBoost"]:
    col = "CNN" if name.startswith("CNN") else name
    prob = df[col].values
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, pred), 3),
        "Sensitivity": round(tp / (tp + fn), 3),
        "Specificity": round(tn / (tn + fp), 3),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_true, prob), 3),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(1)
    plt.plot(fpr, tpr, label=name + " (AUC=" + str(round(roc_auc_score(y_true, prob), 2)) + ")")
    if name.startswith("CNN"):
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, display_labels=["Healthy", "Glaucoma"], cmap="Blues")
        plt.title("CNN - Confusion Matrix")
        plt.savefig(os.path.join(base, "results", "cm_cnn.png"), dpi=150)
        plt.close()

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curves: Image vs Tabular Models (Test Set)")
plt.legend()
plt.savefig(os.path.join(base, "results", "roc_image_vs_tabular.png"), dpi=150)

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv(os.path.join(base, "results", "test_results_all.csv"), index=False)
df.to_csv(os.path.join(base, "results", "test_probabilities.csv"), index=False)
print("\nSaved results/test_results_all.csv, test_probabilities.csv, cm_cnn.png, roc_image_vs_tabular.png")