# src/model.py

from pathlib import Path
import joblib
import numpy as np

# ---------------------------------------------------
# SAFE MODEL LOADER
# ---------------------------------------------------

def load_bundle(path="models/final_model.pkl"):
    """
    Loads the trained ensemble model safely.
    Returns None if file missing.
    """
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Model file not found at: {path}")
        return None

    try:
        bundle = joblib.load(path)
        return bundle
    except Exception as e:
        print(f"[ERROR] Failed to load model bundle: {e}")
        return None


# ---------------------------------------------------
# ENSEMBLE PREDICTOR
# ---------------------------------------------------

def predict_ensemble(X, bundle_path="models/final_model.pkl"):
    """
    Takes a feature DataFrame (X) and returns an ensemble probability score.
    Internally loads the model bundle.

    Models expected in bundle:
    - logistic
    - rf
    - mlp
    - lgbm (optional)
    """

    bundle = load_bundle(bundle_path)

    if bundle is None:
        raise FileNotFoundError("Model bundle could not be loaded. Check model path.")

    # Ensure X is 2D (DataFrame or array)
    if not hasattr(X, "shape"):
        raise ValueError("X must be a DataFrame or 2D array of features.")

    # Collect model outputs
    preds = []

    # Logistic Regression Probability
    if "logistic" in bundle:
        preds.append(bundle["logistic"].predict_proba(X)[:, 1])
    else:
        preds.append(np.zeros(len(X)))

    # Random Forest Probability
    if "rf" in bundle:
        preds.append(bundle["rf"].predict_proba(X)[:, 1])
    else:
        preds.append(np.zeros(len(X)))

    # MLP Probability
    if "mlp" in bundle:
        preds.append(bundle["mlp"].predict_proba(X)[:, 1])
    else:
        preds.append(np.zeros(len(X)))

    # LightGBM Prediction
    if "lgbm" in bundle:
        try:
            preds.append(bundle["lgbm"].predict(X))
        except Exception:
            preds.append(np.zeros(len(X)))
    else:
        preds.append(np.zeros(len(X)))

    # Final average ensemble
    final_prediction = np.mean(preds, axis=0)

    return final_prediction
