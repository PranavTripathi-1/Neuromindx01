import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from datetime import datetime

# ============================================================
#                FILE PATHS & CONSTANTS
# ============================================================

os.makedirs("data", exist_ok=True)

USER_FILE = "data/user_assessments.csv"
RANDOM_FILE = "data/random_assessment_data.csv"

USER_COLUMNS = ["user_id", "assessment_type", "score", "created_at"]


# ============================================================
#                SAFE CSV LOADING FUNCTIONS
# ============================================================

def load_user_data():
    """Loads user assessment CSV safely. Creates empty file if missing."""
    if not os.path.exists(USER_FILE) or os.path.getsize(USER_FILE) == 0:
        df = pd.DataFrame(columns=USER_COLUMNS)
        df.to_csv(USER_FILE, index=False)
        return df
    return pd.read_csv(USER_FILE)


def generate_random_dataset():
    """Creates random dataset ONLY ONCE to compare user with random population."""
    if not os.path.exists(RANDOM_FILE):
        random_df = pd.DataFrame({
            "user_id": np.random.randint(1, 300, 150),
            "assessment_type": np.random.choice(["depression", "anxiety", "stress"], 150),
            "score": np.random.randint(0, 30, 150),
            "created_at": pd.date_range(start="2024-01-01", periods=150).strftime("%Y-%m-%d")
        })
        random_df.to_csv(RANDOM_FILE, index=False)

    return pd.read_csv(RANDOM_FILE)


# Load datasets now
user_df = load_user_data()
random_df = generate_random_dataset()


# ============================================================
#                FUNCTION TO SAVE REAL ASSESSMENT
# ============================================================

def add_user_assessment(user_id, assessment_type, score):
    """Call this function from Assessment page to store results."""
    new_entry = {
        "user_id": user_id,
        "assessment_type": assessment_type,
        "score": score,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = load_user_data()
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(USER_FILE, index=False)

    return True


# ============================================================
#                         UI / DASHBOARD
# ============================================================

st.title("📊 NeuromindX Dashboard")
st.markdown("""
### Welcome to your mental health assessment dashboard  
This dashboard compares your assessment scores with a general population dataset.
""")

# ============================================================
#                    DISPLAY USER DATA
# ============================================================

st.subheader("🧑‍💻 Your Assessment History")

if len(user_df) == 0:
    st.warning("You have not completed any assessments yet.")
else:
    st.dataframe(user_df)

    # --- Line Chart (Your Scores Over Time)
    chart_df = user_df.copy()
    chart_df["created_at"] = pd.to_datetime(chart_df["created_at"])

    fig = px.line(chart_df, x="created_at", y="score", color="assessment_type",
                title="📈 Your Score Trend Over Time",
                markers=True)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#                DISPLAY RANDOM COMPARISON DATA
# ============================================================

st.subheader("📊 Population-Level Comparison Dataset")
st.dataframe(random_df)

# Comparison bar chart
fig2 = px.bar(
    random_df.groupby("assessment_type")["score"].mean().reset_index(),
    x="assessment_type",
    y="score",
    title="📌 Average Mental Health Scores in Random Population",
)
st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#                COMPARE USER vs POPULATION
# ============================================================

st.subheader("⚖ Your Score vs Population Average")

if len(user_df) > 0:
    latest = user_df.iloc[-1]
    user_type = latest["assessment_type"]
    user_score = latest["score"]

    population_avg = random_df[random_df["assessment_type"] == user_type]["score"].mean()

    st.success(f"Latest Assessment: **{user_type.capitalize()}**")
    st.info(f"📌 Your Score: **{user_score}**")
    st.info(f"📊 Population Average: **{round(population_avg, 2)}**")

    comparison_df = pd.DataFrame({
        "Category": ["Your Score", "Population Avg"],
        "Score": [user_score, population_avg]
    })

    fig3 = px.bar(comparison_df, x="Category", y="Score", title="🎯 Comparison Result")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Complete an assessment to view comparison.")


# ============================================================
#         A BUTTON TO REFRESH RANDOM DATA (OPTIONAL)
# ============================================================

with st.expander("🔄 Regenerate Random Dataset"):
    if st.button("Generate New Random Data"):
        os.remove(RANDOM_FILE)
        generate_random_dataset()
        st.success("New random dataset generated! Refresh the page.")


