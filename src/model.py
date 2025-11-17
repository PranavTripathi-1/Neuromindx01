# src/model.py
from pathlib import Path
import joblib

def load_bundle(path="models/ensemble.joblib"):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(path)

def predict_ensemble(bundle, X):
    p1 = bundle["logistic"].predict_proba(X)[:,1]
    p2 = bundle["rf"].predict_proba(X)[:,1]
    p3 = bundle["mlp"].predict_proba(X)[:,1]
    try:
        p4 = bundle["lgbm"].predict(X)
    except Exception:
        p4 = 0*p1
    return (p1 + p2 + p3 + p4) / 4.0
