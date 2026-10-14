import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

# 1. clinical data + CNN probability for each eye
train = pd.read_csv(os.path.join(base, "train.csv")).merge(
    pd.read_csv(os.path.join(base, "cnn_oof_train.csv"))[["ID", "Eye", "CNN_prob"]],
    on=["ID", "Eye"])
test = pd.read_csv(os.path.join(base, "test.csv")).merge(
    pd.read_csv(os.path.join(base, "cnn_test.csv"))[["ID", "Eye", "CNN_prob"]],
    on=["ID", "Eye"])
print("Train rows:", len(train), "| Test rows:", len(test), "(should be 336 and 84)\n")

clinical = [c for c in train.columns if c not in ["ID", "Target", "CNN_prob"]]
y = train["Target"]
groups = train["ID"]

def make_pipe(model, smote=False):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)

candidates = {
    "Age only (LR)": (["Age"], make_pipe(
        LogisticRegression(class_weight="balanced", max_iter=1000))),
    "Fusion: clinical + CNN (LR) [primary]": (clinical + ["CNN_prob"], make_pipe(
        LogisticRegression(class_weight="balanced", max_iter=1000))),
    "Fusion: clinical + CNN (XGBoost)": (clinical + ["CNN_prob"], make_pipe(
        XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      eval_metric="logloss", random_state=42), smote=True)),
}

# 2. cross-validation on the TRAINING set (same patient-level folds for all)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(train, y, groups))
scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]

rows = []
def add_row(name, d):
    row = {"Model": name}
    for m in scoring:
        row[m] = "%.3f ± %.3f" % (np.mean(d[m]), np.std(d[m]))
    rows.append(row)

# CNN alone, scored fold by fold on its out-of-fold probabilities
d = {m: [] for m in scoring}
for tr_i, va_i in splits:
    yy, pp = y.iloc[va_i], train["CNN_prob"].iloc[va_i]
    pred = (pp > 0.5).astype(int)
    d["accuracy"].append(accuracy_score(yy, pred))
    d["recall"].append(recall_score(yy, pred))
    d["precision"].append(precision_score(yy, pred, zero_division=0))
    d["f1"].append(f1_score(yy, pred))
    d["roc_auc"].append(roc_auc_score(yy, pp))
add_row("CNN alone", d)

for name, (cols, pipe) in candidates.items():
    res = cross_validate(pipe, train[cols], y, cv=splits, scoring=scoring)
    add_row(name, {m: res["test_" + m] for m in scoring})

print("=== Cross-validation on the training set ===")
print(pd.DataFrame(rows).to_string(index=False))

# 3. final evaluation on the TEST set (once)
y_test = test["Target"]
probs = {"CNN alone": test["CNN_prob"].values}
for name, (cols, pipe) in candidates.items():
    pipe.fit(train[cols], y)
    probs[name] = pipe.predict_proba(test[cols])[:, 1]

out = []
plt.figure(figsize=(6.5, 5))
for name, p in probs.items():
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    out.append({"Model": name,
                "Accuracy": round(accuracy_score(y_test, pred), 3),
                "Sensitivity": round(tp / (tp + fn), 3),
                "Specificity": round(tn / (tn + fp), 3),
                "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "F1": round(f1_score(y_test, pred), 3),
                "ROC-AUC": round(roc_auc_score(y_test, p), 3),
                "TP": tp, "TN": tn, "FP": fp, "FN": fn})
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label="%s (AUC=%.2f)" % (name.split(" [")[0], roc_auc_score(y_test, p)))

print("\n=== Test set (%d eyes, %d glaucoma) ===" % (len(y_test), y_test.sum()))
print(pd.DataFrame(out).to_string(index=False))

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Image vs Clinical vs Fusion (Test Set)")
plt.legend(fontsize=7)
os.makedirs(os.path.join(base, "results"), exist_ok=True)
plt.savefig(os.path.join(base, "results", "roc_fusion.png"), dpi=150, bbox_inches="tight")
pd.DataFrame(out).to_csv(os.path.join(base, "results", "test_results_fusion.csv"), index=False)
print("\nSaved results/test_results_fusion.csv and results/roc_fusion.png")