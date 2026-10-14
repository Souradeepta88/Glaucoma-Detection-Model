import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(base, "results")

# ---------- 1. ONE results table (test set) ----------
tab = pd.read_csv(os.path.join(res, "test_results.csv"))          # Step 7: SVM, RF, XGBoost
fus = pd.read_csv(os.path.join(res, "test_results_fusion.csv"))   # CNN, age only, fusion
tab["Model"] = tab["Model"].apply(lambda m: m + " (clinical)")
fus["Model"] = fus["Model"].str.replace(" [primary]", "", regex=False)
allres = pd.concat([tab, fus], ignore_index=True)
allres = allres.sort_values("ROC-AUC", ascending=False)

print("=== All models on the test set (84 eyes, 19 glaucoma), sorted by AUC ===")
print(allres.to_string(index=False))
allres.to_csv(os.path.join(res, "final_results_table.csv"), index=False)

# bar chart
plt.figure(figsize=(8, 4.5))
colors = ["#2a9d8f" if ("CNN" in m or "Fusion" in m) else "#999999" for m in allres["Model"]]
plt.barh(allres["Model"], allres["ROC-AUC"], color=colors)
plt.gca().invert_yaxis()
plt.axvline(0.5, color="k", linestyle="--", linewidth=1)
plt.xlim(0.4, 0.95)
plt.xlabel("ROC-AUC (test set)")
plt.title("Test ROC-AUC by model (dashed line = random guess)")
for i, v in enumerate(allres["ROC-AUC"]):
    plt.text(v + 0.005, i, "%.3f" % v, va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(res, "final_auc_comparison.png"), dpi=150)

# ---------- 2. Age check ----------
tr = pd.read_csv(os.path.join(base, "train.csv")).merge(
    pd.read_csv(os.path.join(base, "cnn_oof_train.csv"))[["ID", "Eye", "CNN_prob"]], on=["ID", "Eye"])
te = pd.read_csv(os.path.join(base, "test.csv")).merge(
    pd.read_csv(os.path.join(base, "cnn_test.csv"))[["ID", "Eye", "CNN_prob"]], on=["ID", "Eye"])
d = pd.concat([tr, te], ignore_index=True)

print("\n=== Age check (all 420 eyes, unseen-eye CNN predictions) ===")
rho, _ = spearmanr(d["Age"], d["CNN_prob"])
print("Correlation between CNN probability and age (Spearman): %.2f" % rho)
print("  healthy eyes:  %.2f" % spearmanr(d[d.Target == 0]["Age"], d[d.Target == 0]["CNN_prob"])[0])
print("  glaucoma eyes: %.2f" % spearmanr(d[d.Target == 1]["Age"], d[d.Target == 1]["CNN_prob"])[0])

old = d[d["Age"] >= 50]
print("\nEyes aged 50+: %d (%d glaucoma)" % (len(old), old["Target"].sum()))
print("  AUC of age alone:  %.3f" % roc_auc_score(old["Target"], old["Age"]))
print("  AUC of CNN:        %.3f" % roc_auc_score(old["Target"], old["CNN_prob"]))

print("\nAUC inside narrow age bands (age cannot separate classes much here):")
for lo, hi in [(50, 64), (65, 74), (75, 100)]:
    b = d[(d["Age"] >= lo) & (d["Age"] <= hi)]
    n1, n0 = int(b["Target"].sum()), int((b["Target"] == 0).sum())
    if n1 >= 5 and n0 >= 5:
        print("  age %d-%d: %d healthy, %d glaucoma | CNN AUC %.3f | age AUC %.3f"
              % (lo, hi, n0, n1, roc_auc_score(b["Target"], b["CNN_prob"]),
                 roc_auc_score(b["Target"], b["Age"])))
    else:
        print("  age %d-%d: %d healthy, %d glaucoma | too few eyes to score" % (lo, hi, n0, n1))

print("\nSaved results/final_results_table.csv and results/final_auc_comparison.png")