import streamlit as st
from src.features import extract_features
from src.model import load_model, predict

st.title("🤖 Model Predictions")

user_input = st.text_area("Enter your thoughts here:")
if st.button("Predict"):
    model = load_model()
    features = extract_features(user_input)
    prediction = predict(model, features)
    st.success(f"Predicted Label: {prediction}")
