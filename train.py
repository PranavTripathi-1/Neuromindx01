# train.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import joblib
from src.features import build_feature_dataframe

def create_synthetic(n=1200, seed=42):
    np.random.seed(seed)
    rows = []
    for i in range(n):
        age = int(np.clip(np.random.normal(30, 8), 16, 80))
        symptom = float(np.clip(np.random.beta(2,5) + 0.12*(age>45), 0, 1))
        label = int(symptom > 0.6)
        rts = list(np.random.lognormal(mean=0.5 + 0.8*symptom, sigma=0.22, size=6))
        if symptom < 0.3:
            text = "I am feeling okay, sleeping well and energetic."
        elif symptom < 0.6:
            text = "I am sometimes down, sleep is irregular, energy fluctuates."
        else:
            text = "I feel low and anxious often, can't sleep or focus."
        rows.append({
            "participant_id": f"sub_{i:06d}",
            "age": age,
            "text_response": text,
            "reaction_times": rts,
            "audio_bytes": None,
            "label": label
        })
    return pd.DataFrame(rows)

def main_train(n=1200, out_path="models/ensemble.joblib"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = create_synthetic(n=n)
    # Build features and fit TF-IDF inside build_feature_dataframe with fit
    feat_df, tfidf_vect = build_feature_dataframe(df, tfidf_vect=None, fit_tfidf=True)
    X = feat_df.values
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Logistic
    log = LogisticRegression(max_iter=500)
    log.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    rf.fit(X_train, y_train)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)
    mlp.fit(X_train, y_train)

    # LightGBM (train on full train data)
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {"objective":"binary","metric":"auc","verbosity":-1}
    lgbm = lgb.train(params, dtrain, num_boost_round=200)

    bundle = {
        "logistic": log,
        "rf": rf,
        "mlp": mlp,
        "lgbm": lgbm,
        "tfidf_vect": tfidf_vect,
        "feature_columns": feat_df.columns.tolist()
    }
    joblib.dump(bundle, out_path)
    print("Saved ensemble to", out_path)
    return bundle

if __name__ == "__main__":
    main_train()
