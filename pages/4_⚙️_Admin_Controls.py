import streamlit as st
import os

st.title("⚙️ Admin Controls")

if st.button("Show Models"):
    st.write("Available models:")
    for model_file in os.listdir("data/models"):
        st.write(f"- {model_file}")
