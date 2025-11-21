# generate_models_sklearn13.py

import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except:
    lgbm_available = False

from src.features import extract_features

# ==============================================================
# 1. Generate synthetic dataset
# ==============================================================

def gen_text(label):
    pos = [
        "I feel good today and relaxed.",
        "I am happy and calm.",
        "Everything feels normal and fine.",
    ]
    neg = [
        "I feel depressed and tired.",
        "I'm anxious and stressed all day.",
        "I can't sleep, low mood and sad.",
    ]
    neu = [
        "I went for a walk.",
        "I did some chores today.",
        "It was a normal day.",
    ]

    base = random.choice(pos + neu*2) if label == 0 else random.choice(neg + neu)
    return base + random.choice(["", " I don't know why.", " It's been this way."])


print("Generating data...")
rows = []
for _ in range(1500):
    lbl = 1 if random.random() < 0.5 else 0
    rows.append({"text": gen_text(lbl), "label": lbl})

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/training_data.csv", index=False)


# ==============================================================
# 2. Extract features using YOUR extractor
# ==============================================================

print("Extracting features...")
X = pd.DataFrame([extract_features(t) for t in df["text"]])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================================================
# 3. Train models using sklearn 1.3.0 (YOUR system)
# ==============================================================

print("Training Logistic Regression...")
logistic = LogisticRegression(max_iter=2000)
logistic.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print("Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)
mlp.fit(X_train, y_train)

if lgbm_available:
    print("Training LightGBM...")
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
else:
    print("LightGBM not installed → fallback RF")
    lgbm = rf


# ==============================================================
# 4. Save models properly for sklearn 1.3.0
# ==============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(logistic, "models/logistic.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(mlp, "models/mlp.pkl")
joblib.dump(lgbm, "models/lgbm.pkl")

bundle = {
    "logistic": logistic,
    "rf": rf,
    "mlp": mlp,
    "lgbm": lgbm
}

joblib.dump(bundle, "models/ensemble.pkl")
joblib.dump(bundle, "models/final_model.pkl")

print("\n✔ All models saved in /models using sklearn 1.3.0")
