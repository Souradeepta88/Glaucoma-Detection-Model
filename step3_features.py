import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base, "clean_data.csv"))

# 3a. Is "missing" related to the diagnosis? (checking for leakage)
feats = data.drop(columns=["ID", "Eye", "Diagnosis"])
print("% missing per column, by diagnosis (0=healthy, 1=glaucoma, 2=suspect):")
print((feats.isnull().groupby(data["Diagnosis"]).mean() * 100).round(1).T)
print()

# 3b. Keep only healthy (0) and glaucoma (1)
data = data[data["Diagnosis"].isin([0, 1])].copy()
data["Target"] = data["Diagnosis"]

# 3c. Turn Eye into a number
data["Eye"] = data["Eye"].map({"OD": 0, "OS": 1})

# 3d. Combine IOP columns into one, preferring Pneumatic over Perkins.
data["IOP"] = data["IOP_Pneumatic"].fillna(data["IOP_Perkins"])

# 3e. Drop leaky columns
data = data.drop(columns=["VF_MD", "IOP_Perkins", "IOP_Pneumatic", "Diagnosis"])

# 3f. Final feature list
features = ["Age", "Gender", "Sphere", "Cylinder", "Axis", "Lens_Status",
            "IOP", "Pachymetry", "Axial_Length", "Eye"]

model_data = data[["ID"] + features + ["Target"]]

print("Model data shape:", model_data.shape)
print()
print("Target counts (0=healthy, 1=glaucoma):")
print(model_data["Target"].value_counts())
print()
print("Remaining missing values:")
print(model_data.isnull().sum())

model_data.to_csv(os.path.join(base, "model_data.csv"), index=False)
print("\nSaved model_data.csv")