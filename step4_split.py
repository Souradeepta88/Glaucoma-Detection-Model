import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold

base = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base, "model_data.csv"))

# 5 folds -> take one fold as the test set (~20%)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(data, data["Target"], groups=data["ID"]))

train = data.iloc[train_idx]
test = data.iloc[test_idx]

# 1. no patient should appear in both sets
overlap = set(train["ID"]) & set(test["ID"])
print("Patients in both train and test (must be 0):", len(overlap))
print()

# 2. sizes and class balance
print("Train rows:", len(train), "| patients:", train["ID"].nunique())
print("Test rows: ", len(test), "| patients:", test["ID"].nunique())
print()
print("Train glaucoma %:", round(train["Target"].mean() * 100, 1))
print("Test glaucoma %: ", round(test["Target"].mean() * 100, 1))
print()
print("Train target counts:\n", train["Target"].value_counts())
print("Test target counts:\n", test["Target"].value_counts())

# 3. save for later steps
train.to_csv(os.path.join(base, "train.csv"), index=False)
test.to_csv(os.path.join(base, "test.csv"), index=False)

split_ids = pd.concat([
    train[["ID"]].drop_duplicates().assign(Set="train"),
    test[["ID"]].drop_duplicates().assign(Set="test"),
])
split_ids.to_csv(os.path.join(base, "split_ids.csv"), index=False)
print("\nSaved train.csv, test.csv, split_ids.csv")