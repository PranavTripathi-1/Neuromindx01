# src/explainability.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def shap_summary_plot_lgb(lgbm_model, X, feature_names, max_display=20):
    try:
        explainer = shap.TreeExplainer(lgbm_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_vals, pd.DataFrame(X, columns=feature_names), show=False, max_display=max_display)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def top_shap_table(lgbm_model, X, feature_names, top_k=10):
    try:
        explainer = shap.TreeExplainer(lgbm_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({"feature":feature_names, "mean_abs_shap":mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["feature","mean_abs_shap"])
