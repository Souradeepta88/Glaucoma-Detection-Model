import pandas as pd
import numpy as np
import os
import joblib
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
test = pd.read_csv(os.path.join(base, "test.csv"))
X_test = test.drop(columns=["ID", "Target"])

pipe = joblib.load(os.path.join(base, "models", "xgboost.joblib"))

# run the data through the same preprocessing the model saw
imputed = pipe.named_steps["impute"].transform(X_test)
scaled = pipe.named_steps["scale"].transform(imputed)
model = pipe.named_steps["model"]

imputed_df = pd.DataFrame(imputed, columns=X_test.columns)   # original units, for display
scaled_df = pd.DataFrame(scaled, columns=X_test.columns)     # what the model actually sees

# SHAP values computed natively by XGBoost (same TreeSHAP algorithm)
dmat = xgb.DMatrix(scaled_df, feature_names=list(X_test.columns))
contribs = model.get_booster().predict(dmat, pred_contribs=True)
shap_values = contribs[:, :-1]   # last column is the bias term, so drop it

# ranking of features by average impact
importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
importance = importance.sort_values(ascending=False)
print("Mean |SHAP| per feature (higher = more important):")
print(importance.round(4).to_string())

os.makedirs(os.path.join(base, "results"), exist_ok=True)

# 1. bar chart of importance
plt.figure()
shap.summary_plot(shap_values, imputed_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(base, "results", "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. beeswarm: direction of each feature's effect (red = high value)
plt.figure()
shap.summary_plot(shap_values, imputed_df, show=False)
plt.tight_layout()
plt.savefig(os.path.join(base, "results", "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved results/shap_bar.png and results/shap_beeswarm.png")