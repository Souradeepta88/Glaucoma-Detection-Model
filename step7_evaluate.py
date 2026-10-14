import pandas as pd
import os
import joblib
import matplotlib
matplotlib.use("Agg")   # lets WSL save plots without a display
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

base = os.path.dirname(os.path.abspath(__file__))
test = pd.read_csv(os.path.join(base, "test.csv"))

X_test = test.drop(columns=["ID", "Target"])
y_test = test["Target"]

files = {"SVM": "svm.joblib",
         "Random Forest": "random_forest.joblib",
         "XGBoost": "xgboost.joblib"}

os.makedirs(os.path.join(base, "results"), exist_ok=True)

rows = []
plt.figure(figsize=(6, 5))

for name, f in files.items():
    model = joblib.load(os.path.join(base, "models", f))
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Sensitivity": round(sensitivity, 3),
        "Specificity": round(specificity, 3),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "F1": round(f1_score(y_test, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 3),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    })

    # confusion matrix picture
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Healthy", "Glaucoma"], cmap="Blues")
    plt.title(name + " - Confusion Matrix")
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(base, "results", "cm_" + fname + ".png"), dpi=150)
    plt.close()

    # add this model's line to the shared ROC plot
    plt.figure(1)
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name + " (AUC=" + str(round(roc_auc_score(y_test, prob), 2)) + ")")

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curves on Test Set")
plt.legend()
plt.savefig(os.path.join(base, "results", "roc_curves.png"), dpi=150)

results = pd.DataFrame(rows)
print("Test set: %d rows (%d healthy, %d glaucoma)\n" %
      (len(y_test), (y_test == 0).sum(), (y_test == 1).sum()))
print(results.to_string(index=False))
results.to_csv(os.path.join(base, "results", "test_results.csv"), index=False)
print("\nSaved results/test_results.csv, roc_curves.png and 3 confusion matrices")