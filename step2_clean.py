import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base, "PapilaDB", "ClinicalData")

# column names (the 3 junk rows at the top are skipped)
cols = ["ID", "Age", "Gender", "Diagnosis",
        "Sphere", "Cylinder", "Axis",
        "Lens_Status", "IOP_Pneumatic", "IOP_Perkins",
        "Pachymetry", "Axial_Length", "VF_MD"]

def load_eye(filename, eye):
    df = pd.read_excel(os.path.join(folder, filename), header=None, skiprows=3)
    df.columns = cols
    df["Eye"] = eye
    return df

od = load_eye("patient_data_od.xlsx", "OD")   # right eye
os_ = load_eye("patient_data_os.xlsx", "OS")  # left eye

# combine both eyes: one row per eye
data = pd.concat([od, os_], ignore_index=True)

# convert everything except ID and Eye to numbers
for c in cols:
    if c not in ["ID"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

# drop rows with no diagnosis (nothing to learn from)
data = data.dropna(subset=["Diagnosis"])
data["Diagnosis"] = data["Diagnosis"].astype(int)

print("Final shape:", data.shape)
print()
print(data.head(8))
print()
data.info()
print()
print("Diagnosis counts (0=healthy, 1=glaucoma, 2=suspect):")
print(data["Diagnosis"].value_counts())
print()
print("Missing values:\n", data.isnull().sum())
print()
print("Unique patients:", data["ID"].nunique())

data.to_csv(os.path.join(base, "clean_data.csv"), index=False)
print("\nSaved clean_data.csv")