import streamlit as st
from src.assessments_utils import run_assessment

st.set_page_config(page_title="Assessments — Neuropsy", page_icon="🧠", layout="centered")
st.title("Neuropsychiatric Early Screening")
run_assessment()
