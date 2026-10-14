import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

base = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base, "train.csv"))

X = train.drop(columns=["ID", "Target"])
y = train["Target"]
groups = train["ID"]

print("Accuracy of always predicting 'healthy':",
      round((y == 0).mean() * 100, 1), "%\n")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(use_smote=False, class_weight=None):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", RandomForestClassifier(
        n_estimators=200, class_weight=class_weight, random_state=42)))
    return Pipeline(steps)

strategies = {
    "No fix (baseline)": make_pipe(),
    "Class weights":     make_pipe(class_weight="balanced"),
    "SMOTE":             make_pipe(use_smote=True),
}

scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]

rows = []
for name, pipe in strategies.items():
    res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring)
    row = {"Strategy": name}
    for m in scoring:
        scores = res["test_" + m]
        row[m] = f"{scores.mean():.3f} ± {scores.std():.3f}"
    rows.append(row)

print(pd.DataFrame(rows).to_string(index=False))