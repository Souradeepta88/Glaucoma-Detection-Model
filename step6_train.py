import pandas as pd
import os
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

base = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base, "train.csv"))

X = train.drop(columns=["ID", "Target"])
y = train["Target"]
groups = train["ID"]

def make_pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model),
    ])

models = {
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=42),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]

rows = []
for name, model in models.items():
    pipe = make_pipe(model)
    res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring)
    row = {"Model": name}
    for m in scoring:
        s = res["test_" + m]
        row[m] = f"{s.mean():.3f} ± {s.std():.3f}"
    rows.append(row)

print(pd.DataFrame(rows).to_string(index=False))

# train the final version of each model on ALL training data and save it
os.makedirs(os.path.join(base, "models"), exist_ok=True)
for name, model in models.items():
    pipe = make_pipe(model)
    pipe.fit(X, y)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(base, "models", fname))
    print("Saved models/" + fname)