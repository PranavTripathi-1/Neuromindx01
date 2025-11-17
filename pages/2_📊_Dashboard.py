import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_utils import get_summary_metrics, get_radar_chart, get_recommendations

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Mental Health Insights Dashboard")

st.markdown("""
Welcome to your **Personal Wellness Dashboard**.  
Here you’ll see a summary of your recent assessments and any early-warning signals for neuropsychiatric conditions (like depression, anxiety, ADHD, or mild cognitive issues).
""")

# Load user data
try:
    df = pd.read_csv("data/user_assessments.csv")
except FileNotFoundError:
    st.warning("⚠️ No assessment data found. Please complete an assessment first!")
    st.stop()

summary = get_summary_metrics(df)

# --- Top cards
st.markdown("### 🧠 Overall Mental Health Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mood Stability", f"{summary['mood_stability']} / 10")
col2.metric("Stress Control", f"{summary['stress_control']} / 10")
col3.metric("Sleep Quality", f"{summary['sleep_quality']} / 10")
col4.metric("Energy Levels", f"{summary['energy']} / 10")

st.markdown("---")

# --- Overall score
st.subheader("💫 Overall Health Index")
st.progress(summary["wellness_score"] / 100)
st.write(f"**Your Mental Wellness Score:** {summary['wellness_score']:.1f}/100")

# --- Radar Chart
st.markdown("### 📈 Pattern Overview")
radar_fig = get_radar_chart(summary)
st.plotly_chart(radar_fig, use_container_width=True)

# --- Risk Analysis Bar Chart
risk_levels = pd.DataFrame({
    "Domain": ["Depression Risk", "Anxiety Risk", "ADHD-Like Patterns", "Cognitive Decline Risk"],
    "Probability (%)": [
        summary["depression_risk"],
        summary["anxiety_risk"],
        summary["adhd_risk"],
        summary["cognitive_risk"]
    ]
})
bar_fig = px.bar(
    risk_levels,
    x="Domain",
    y="Probability (%)",
    color="Probability (%)",
    color_continuous_scale="RdYlGn_r",
    title="🧩 Risk Prediction Overview"
)
st.plotly_chart(bar_fig, use_container_width=True)

# --- Recommendations
st.markdown("### 🌿 Personalized Recommendations")
for rec in get_recommendations(summary):
    st.success(f"💡 {rec}")

st.markdown("---")
st.markdown("If any of your risk levels are above 70%, it’s advisable to consult a licensed mental-health professional for a detailed evaluation.")
