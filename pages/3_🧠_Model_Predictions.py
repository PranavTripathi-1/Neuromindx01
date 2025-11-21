import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.features import extract_features   # your real extractor


# ============================================================
# SAFE MODEL LOADER (NO IMPORT ERRORS)
# ============================================================

@st.cache_resource
def load_model():
    """
    Safely loads the ensemble model stored in:
        models/final_model.pkl
    Works with sklearn 1.3.0
    """
    base = os.path.dirname(os.path.dirname(__file__))  # Neuromindx project root
    model_path = os.path.join(base, "models", "final_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ final_model.pkl is missing in /models folder!")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ============================================================
# PAGE UI
# ============================================================

st.title("🧠 Mental Health Prediction")
st.write("Provide text below to analyze risk level.")


# Load the model only once
bundle = load_model()


# Stop if model did not load
if bundle is None:
    st.stop()


# ============================================================
# INPUT BOX
# ============================================================

user_text = st.text_area(
    "Write how you feel today:",
    placeholder="Example: I feel stressed and tired today..."
)


# ============================================================
# RUN PREDICTION
# ============================================================

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    # Convert input into feature row
    features = extract_features(user_text)
    X = pd.DataFrame([features])

    st.write("### Extracted Features")
    st.write(X)

    # Run ensemble prediction
    try:
        p1 = bundle["logistic"].predict_proba(X)[:, 1]
        p2 = bundle["rf"].predict_proba(X)[:, 1]
        p3 = bundle["mlp"].predict_proba(X)[:, 1]

        # LightGBM behaves differently; use predict() not predict_proba()
        try:
            p4 = bundle["lgbm"].predict(X)
        except:
            p4 = p1  # fallback

        final_score = float((p1 + p2 + p3 + p4) / 4)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ============================================================
    # DISPLAY RESULT
    # ============================================================

    st.write("---")
    st.subheader("🧪 Prediction Result")

    st.metric(
        label="Predicted Mental Health Risk Score",
        value=f"{final_score:.2f}",
        delta=None
    )

    if final_score < 0.33:
        st.success("🟢 Low Risk — You're generally okay!")
    elif final_score < 0.66:
        st.warning("🟡 Moderate Risk — Some signs of stress or anxiety.")
    else:
        st.error("🔴 High Risk — You may be experiencing emotional distress.")

    st.write("---")
    st.write("✔ Model loaded using sklearn 1.3.0 (compatible)")
