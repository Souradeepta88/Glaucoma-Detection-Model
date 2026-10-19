import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from fundus_dataset import eval_tf

base = os.path.dirname(os.path.abspath(__file__))

# ---------- INPUT: change these for the eye you want to predict ----------
IMAGE_PATH = os.path.join(base, "PapilaDB", "FundusImages", "RET004OD.jpg")
clinical = {
    "Age": 58, "Gender": 1, "Sphere": 1.5, "Cylinder": np.nan, "Axis": np.nan,
    "Lens_Status": np.nan, "IOP": 19, "Pachymetry": 501, "Axial_Length": 23.06, "Eye": 0,
}   # unknown values can be np.nan; the saved pipeline fills them with training medians

# ---------- 1. CNN probability: average of the five fold networks ----------
def load_cnn(path):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

img = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224), Image.LANCZOS)
x = eval_tf(img).unsqueeze(0)

probs = []
with torch.no_grad():
    for k in range(1, 6):
        net = load_cnn(os.path.join(base, "models", f"resnet18_fold{k}.pt"))
        probs.append(torch.softmax(net(x), dim=1)[0, 1].item())
cnn_prob = float(np.mean(probs))

# ---------- 2. Fusion model ----------
bundle = joblib.load(os.path.join(base, "models", "fusion_lr.joblib"))
row = pd.DataFrame([{**clinical, "CNN_prob": cnn_prob}])[bundle["features"]]
fusion_prob = bundle["pipeline"].predict_proba(row)[0, 1]

print("Image:", os.path.basename(IMAGE_PATH))
print("CNN probability of glaucoma (mean of 5 networks): %.3f" % cnn_prob)
print("Fusion probability of glaucoma:                  %.3f" % fusion_prob)
print("Prediction at 0.5 threshold:", "GLAUCOMA" if fusion_prob > 0.5 else "HEALTHY")
print("Research prototype only, not for clinical use.")