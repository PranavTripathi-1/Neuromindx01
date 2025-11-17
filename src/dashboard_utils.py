import numpy as np
import plotly.graph_objects as go

def get_summary_metrics(df):
    mood = df.iloc[-1, 0]
    sleep = df.iloc[-1, 1]
    stress = df.iloc[-1, 2]
    energy = df.iloc[-1, 3]
    thoughts = df.iloc[-1, 4]
    total = df.iloc[-1, 5]

    wellness_score = round(100 - (total / 50) * 100, 1)
    wellness_score = max(0, min(wellness_score, 100))

    depression_risk = np.clip(100 - (mood * 10), 0, 100)
    anxiety_risk = np.clip(stress * 20, 0, 100)
    adhd_risk = np.clip((10 - energy) * 10, 0, 100)
    cognitive_risk = np.clip(thoughts * 15, 0, 100)

    return {
        "mood_stability": mood,
        "stress_control": 10 - stress,
        "sleep_quality": 10 - sleep,
        "energy": energy,
        "wellness_score": wellness_score,
        "depression_risk": depression_risk,
        "anxiety_risk": anxiety_risk,
        "adhd_risk": adhd_risk,
        "cognitive_risk": cognitive_risk
    }

def get_radar_chart(summary):
    labels = ["Mood", "Stress", "Sleep", "Energy", "Cognition"]
    values = [
        summary["mood_stability"],
        summary["stress_control"],
        summary["sleep_quality"],
        summary["energy"],
        10 - (summary["cognitive_risk"] / 10)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name='User Profile', line_color='royalblue'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False)
    return fig

def get_recommendations(summary):
    recs = []
    if summary["depression_risk"] > 60:
        recs.append("Engage in daily physical activity and journaling to stabilize mood.")
    if summary["anxiety_risk"] > 60:
        recs.append("Practice deep breathing or mindfulness for 5 minutes twice daily.")
    if summary["adhd_risk"] > 60:
        recs.append("Try scheduling short, focused work sprints (Pomodoro technique).")
    if summary["cognitive_risk"] > 60:
        recs.append("Do mental exercises like puzzles, reading, or creative writing.")
    if summary["wellness_score"] > 80:
        recs.append("You’re in a great mental state! Keep maintaining your healthy habits.")
    if not recs:
        recs.append("No major risks detected — keep balancing rest, nutrition, and focus.")
    return recs
