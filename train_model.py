# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.features import extract_features
import os


# ============================================================
# 1. CHECK DATASET
# ============================================================

DATA_PATH = "data/training_data.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "❌ training_data.csv NOT FOUND!\n"
        "Place your dataset at: data/training_data.csv\n\n"
        "Required Columns:\n"
        "- text\n"
        "- label (0 = normal, 1 = mental health risk)"
    )

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(
        "❌ Dataset must contain 'text' and 'label' columns.\n"
        "Example row:\n"
        "text: 'I feel sad and tired'\n"
        "label: 1"
    )


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

print("\nExtracting features...")

feature_rows = []

for txt in df["text"]:
    feature_rows.append(extract_features(txt))

X = pd.DataFrame(feature_rows)
y = df["label"]

print(f"Features shape: {X.shape}")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. TRAIN MODELS
# ============================================================

print("\nTraining Models...")

# 1️⃣ Logistic Regression
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

# 2️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

# 3️⃣ MLP Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600)
mlp.fit(X_train, y_train)

# 4️⃣ LightGBM Model
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

def evaluate(name, model):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))


print("\n============ MODEL PERFORMANCE ============")
evaluate("Logistic Regression", logistic)
evaluate("Random Forest", rf)
evaluate("MLP Neural Net", mlp)
evaluate("LightGBM", lgbm)


# ============================================================
# 6. SAVE ENSEMBLE
# ============================================================

bundle = {
    "logistic": logistic,
    "rf": rf,
    "mlp": mlp,
    "lgbm": lgbm
}

os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/final_model.pkl")

print("\n✔✔✔ FINAL MODEL SAVED: models/final_model.pkl ✔✔✔")
