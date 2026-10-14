import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(base, "model_data.csv"))
print(d.groupby("Target")["Age"].describe().round(1))