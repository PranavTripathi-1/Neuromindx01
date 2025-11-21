import streamlit as st
import pandas as pd
from src.features import extract_features
from src.model import predict_ensemble   # uses improved model loader

st.set_page_config(page_title="Model Predictions", page_icon="🧠")


# ---------------------------------------------------------
#                 PAGE HEADER
# ---------------------------------------------------------

st.title("🧠 AI-Based Neuropsychiatric Prediction")
st.write("""
Provide your symptoms or emotional description, and the AI model will analyze
patterns and predict the likelihood of a neuropsychiatric condition
(depression, anxiety, or stress).
""")


# ---------------------------------------------------------
#             USER INPUT SECTION
# ---------------------------------------------------------

user_text = st.text_area(
    "📝 Describe how you're feeling today:",
    placeholder="Write about your mood, thoughts, sleep, energy, stress, etc...",
    height=180
)

# Action button
if st.button("🔍 Predict Condition"):

    if len(user_text.strip()) < 5:
        st.error("Please enter a longer and meaningful description.")
        st.stop()

    # ---------------------------------------------------------
    #       1. Extract Features from User Text
    # ---------------------------------------------------------

    features = extract_features(user_text)

    # Convert dict → DataFrame with single row
    X = pd.DataFrame([features])

    st.subheader("🧪 Extracted Features")
    st.json(features)

    # ---------------------------------------------------------
    #       2. Run ENSEMBLE MODEL PREDICTION
    # ---------------------------------------------------------

    try:
        score = predict_ensemble(X)[0]    # get first row value
    except FileNotFoundError:
        st.error("❌ Model file missing! Please place your 'final_model.pkl' inside /models folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()


    # ---------------------------------------------------------
    #      3. Interpret the result into human labels
    # ---------------------------------------------------------

    if score < 0.33:
        label = "🟢 Low Risk — Normal / Mild"
        explanation = "Your emotional patterns do not strongly indicate a neuropsychiatric disorder."
    elif 0.33 <= score < 0.66:
        label = "🟡 Moderate Risk — Observe"
        explanation = "There are some signals of emotional imbalance. Monitoring is recommended."
    else:
        label = "🔴 High Risk — Clinical Suggestive"
        explanation = "Strong indicators of psychological distress such as depression, anxiety, or stress."

    # ---------------------------------------------------------
    #      4. Display Result
    # ---------------------------------------------------------

    st.subheader("🎯 Final Prediction Score")
    st.metric("Risk Probability", f"{score:.2f}")

    st.subheader("🧠 Model Interpretation")
    st.success(label)
    st.write(explanation)

    st.info("⚠ This is not a clinical diagnosis. Consult a professional if symptoms persist.")


# ---------------------------------------------------------
#             SIDEBAR INFORMATION
# ---------------------------------------------------------

st.sidebar.header("ℹ Model Info")
st.sidebar.write("""
This model uses a **4-model ensemble**:

- Logistic Regression  
- Random Forest  
- MLP Neural Network  
- LightGBM  

The final prediction is the average probability from all models.
""")
