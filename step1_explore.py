import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base, "PapilaDB", "ClinicalData")

print("ClinicalData path:", folder)
print("Files inside:", os.listdir(folder))
print()

od = pd.read_excel(os.path.join(folder, "patient_data_od.xlsx"))
os_ = pd.read_excel(os.path.join(folder, "patient_data_os.xlsx"))

print("Right eye shape:", od.shape)
print("Left eye shape:", os_.shape)
print()
print(od.head())
print()
print("Columns:", od.columns.tolist())
print()
od.info()
print()
print(od.describe())
print()
print("Missing values:\n", od.isnull().sum())