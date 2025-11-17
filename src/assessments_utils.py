# src/assessments_utils.py
import streamlit as st
import pandas as pd
import time
import uuid
from datetime import datetime

DATA_FILE = "data/user_assessments.csv"

# ---------- Utilities ----------
def save_result(record: dict):
    """Append result to CSV (creates file with header if missing)."""
    df = pd.DataFrame([record])
    try:
        existing = pd.read_csv(DATA_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(DATA_FILE, index=False)

def risk_label(score, thresholds):
    """Map numeric score to label with thresholds sorted ascending list of (threshold,label)."""
    for th, label in thresholds:
        if score <= th:
            return label
    return thresholds[-1][1]

# ---------- Screening instruments ----------
def phq9():
    """PHQ-9 quick implementation. Returns total and item 9 (suicidal ideation)"""
    st.subheader("PHQ-9 — Depression screener (9 items)")
    items = [
        "Little interest/pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling/staying asleep or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things",
        "Moving/speaking slowly OR being fidgety/restless",
        "Thoughts that you would be better off dead or of hurting yourself"  # item 9
    ]
    responses = []
    st.write("For each item, choose how often you've been bothered by the problem over the past 2 weeks.")
    opts = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}
    for i, q in enumerate(items, 1):
        val = st.radio(f"{i}. {q}", options=list(opts.keys()), format_func=lambda x: opts[x], key=f"phq9_{i}")
        responses.append(val)
    total = sum(responses)
    item9 = responses[-1]
    st.write(f"**PHQ-9 total:** {total} (0–27)")
    return total, item9

def gad7():
    """GAD-7 anxiety screener"""
    st.subheader("GAD-7 — Anxiety screener (7 items)")
    items = [
        "Feeling nervous, anxious or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]
    responses = []
    st.write("Choose how often you were bothered by each over the last 2 weeks.")
    opts = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}
    for i, q in enumerate(items, 1):
        val = st.radio(f"{i}. {q}", options=list(opts.keys()), format_func=lambda x: opts[x], key=f"gad7_{i}")
        responses.append(val)
    total = sum(responses)
    st.write(f"**GAD-7 total:** {total} (0–21)")
    return total

def mdq():
    """Mood Disorder Questionnaire (screen for bipolar) simplified: presence of manic symptoms + impairment"""
    st.subheader("Mood Disorder Questionnaire (MDQ) — Bipolar screening")
    st.write("Select the symptoms you've experienced during the same period of time (Yes/No).")
    symptoms = [
        "Overly happy/labile mood",
        "Decreased need for sleep",
        "More talkative/pressure to keep talking",
        "Racing thoughts",
        "Easily distracted",
        "Increase in goal-directed activity",
        "Excessive involvement in risky activities"
    ]
    responses = []
    for i, s in enumerate(symptoms, 1):
        val = st.checkbox(f"{i}. {s}", key=f"mdq_{i}")
        responses.append(int(val))
    impairment = st.selectbox("Did these symptoms occur during the same time period and cause moderate/severe problems?", ["No", "Yes"], key="mdq_impair")
    total_symptoms = sum(responses)
    positive = (total_symptoms >= 7 and impairment == "Yes") or (total_symptoms >= 5 and impairment == "Yes")
    st.write(f"Symptoms checked: {total_symptoms}. MDQ positive screen: {'Yes' if positive else 'No'}")
    return total_symptoms, positive

def pqb():
    """Prodromal Questionnaire - brief (PQ-B) style short psychosis risk screener"""
    st.subheader("Prodromal symptoms screener (brief)")
    st.write("Have you experienced unusual thoughts, perceptual experiences, or suspiciousness recently?")
    items = [
        "Do you feel that others are watching or talking about you?",
        "Have you experienced unusual perceptions (seeing/hearing) that others do not?",
        "Do your thoughts sometimes feel strange or hard to control?",
        "Do you feel confused about whether things you think are real or not?"
    ]
    responses = []
    for i, q in enumerate(items, 1):
        val = st.radio(f"{i}. {q}", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes", key=f"pqb_{i}")
        responses.append(val)
    total = sum(responses)
    st.write(f"PQ-B short total: {total} (higher → more prodromal features)")
    return total

# ---------- Short cognition + motor tasks ----------
def memory_recall():
    """Simple immediate recall: show 5 words briefly then ask to recall."""
    st.subheader("Memory — Immediate recall")
    words = ["apple","penny","river","window","tiger"]
    st.write("You will see a list of 5 simple words for 6 seconds. Try to remember as many as you can.")
    if st.button("Show words (6s)", key="show_words"):
        st.write(" • ".join(words))
        time.sleep(6)
        st.write(" " * 50)  # clear-ish
    recall = st.text_input("Type the words you remember (separate by commas)", key="recall_input")
    # simple scoring: count matched words
    got = 0
    if recall:
        got = sum(1 for w in words if w.lower() in recall.lower())
    st.write(f"Recalled: {got} / {len(words)}")
    return got

def verbal_fluency():
    """Name as many animals in 60 seconds — count approximate by user input split."""
    st.subheader("Verbal fluency (animals in 60s)")
    st.write("When ready, press Start then type the animals you can name separated by commas. You have 60 seconds.")
    if st.button("Start fluency timer", key="vf_start"):
        st.session_state['vf_started'] = True
        st.session_state['vf_start_time'] = time.time()
    count = 0
    if st.session_state.get('vf_started'):
        elapsed = int(time.time() - st.session_state['vf_start_time'])
        if elapsed >= 60:
            st.write("⏱ Time's up! Please submit your answers.")
            st.session_state['vf_started'] = False
        else:
            st.write(f"⏱ Elapsed: {elapsed} / 60 s")
    ans = st.text_input("Type animals (commas) and press Enter when done", key="vf_input")
    if ans:
        items = [a.strip() for a in ans.split(",") if a.strip()]
        count = len(items)
    st.write(f"Animals named: {count}")
    return count

def clock_drawing():
    """Simple clock task: ask user to draw a clock (approx) by describing where hands point.
       We'll ask them to input hour & minute positions as a proxy."""
    st.subheader("Clock task (orientation & visuospatial)")
    st.write("Draw a clock showing 10 past 11: for this quick screen, enter where the hour and minute hands point.")
    hour = st.number_input("Hour hand points to (1-12)", min_value=1, max_value=12, value=11, key="clock_hour")
    minute = st.number_input("Minute hand minutes (0-59)", min_value=0, max_value=59, value=10, key="clock_min")
    # scoring heuristic: hour around 11 and minute around 10 -> correct
    score = 0
    if hour in (10,11,12):
        score += 1
    if minute in (9,10,11):
        score += 1
    st.write(f"Clock task score (0-2): {score}")
    return score

def motor_tapping():
    """Simple finger tapping: user presses a button repeatedly for 10s, we measure count.
       This approximates bradykinesia if very low speed."""
    st.subheader("Motor tapping test (10 seconds)")
    st.write("Tap the button as many times as you can for 10 seconds.")
    if 'tap_count' not in st.session_state:
        st.session_state['tap_count'] = 0
        st.session_state['tap_start'] = None
        st.session_state['tapping'] = False

    def tap():
        if st.session_state['tap_start'] is None:
            st.session_state['tap_start'] = time.time()
            st.session_state['tapping'] = True
        st.session_state['tap_count'] += 1
        # auto-stop at 10s
        if time.time() - st.session_state['tap_start'] >= 10:
            st.session_state['tapping'] = False

    if st.button("Tap!", on_click=tap, key="tap_button"):
        pass

    if st.session_state.get('tap_start'):
        elapsed = time.time() - st.session_state['tap_start']
        st.write(f"Elapsed: {int(elapsed)}s — Taps: {st.session_state['tap_count']}")
        if elapsed >= 10 and st.session_state['tapping'] == False:
            st.write("⏱ Test finished")
    taps = st.session_state.get('tap_count', 0)
    return taps

# ---------- Aggregate assessment ----------
def run_assessment():
    st.markdown("<h2 style='text-align:center;'>🧠 Neuropsychiatric Early-Screen — Multi-domain</h2>",
                unsafe_allow_html=True)
    st.info("This tool screens for early signs of common neuropsychiatric conditions (depression, anxiety, bipolar, psychosis risk, cognitive impairment, Parkinsonian motor signs). It is NOT a diagnosis. See sources/references in the app.")

    # 1. Symptom screeners
    phq_total, phq9_item = phq9()
    gad_total = gad7()
    mdq_total, mdq_positive = mdq()
    pqb_total = pqb()

    # 2. Cognitive tasks
    st.markdown("---")
    st.write("### Quick cognitive & motor tasks (brief proxies)")
    mem_score = memory_recall()
    vf_score = verbal_fluency()
    clock_score = clock_drawing()
    taps = motor_tapping()

    # 3. Scoring heuristics & mapping (simple rule-based)
    # Depression risk (PHQ-9): minimal(0-4), mild(5-9), mod(10-14), mod-severe(15-19), severe(20+)
    dep_label = risk_label(phq_total, [(4,"Minimal"),(9,"Mild"),(14,"Moderate"),(19,"Moderately severe"),(27,"Severe")])
    anx_label = risk_label(gad_total, [(4,"Minimal"),(9,"Mild"),(14,"Moderate"),(21,"Severe")])
    psychosis_risk = "Low"
    if pqb_total >= 2:
        psychosis_risk = "Elevated"
    bipolar_flag = "Positive" if mdq_positive else "Negative"

    # Cognitive quick risk: combine sub-scores
    cog_total = mem_score + (1 if vf_score < 10 else 2) + clock_score  # small heuristic
    if cog_total <= 2:
        cog_label = "Possible Cognitive Concern"
    elif cog_total <= 4:
        cog_label = "Mild Concerns"
    else:
        cog_label = "No major cognitive concerns noted"

    # Motor: tapping speed heuristic (taps in 10 sec): normative values vary; low taps -> flag
    motor_label = "Normal"
    if taps < 15:
        motor_label = "Bradykinesia-like (low taps) — consider evaluation for parkinsonism"

    # 4. Urgent flags
    urgent_messages = []
    if phq9_item >= 1:
        urgent_messages.append("PHQ-9 item 9 flagged — suicidal ideation. If you have active plan or intent, seek immediate help / emergency services.")
    if pqb_total >= 3:
        urgent_messages.append("Several prodromal psychosis features reported — consider urgent psychiatric evaluation.")
    if urgent_messages:
        st.error("⚠️ Urgent flags:")
        for m in urgent_messages:
            st.write(f"- {m}")
        # add immediate resources
        st.markdown("**If you are in immediate danger or feel you might harm yourself, contact local emergency services now.**")

    # 5. Overall summary (rule-based)
    summary = []
    if dep_label in ("Moderate","Moderately severe","Severe"):
        summary.append(("Depression", dep_label))
    if anx_label in ("Moderate","Severe"):
        summary.append(("Anxiety", anx_label))
    if bipolar_flag == "Positive":
        summary.append(("Bipolar disorder (screen)", "Possible — MDQ positive"))
    if psychosis_risk == "Elevated":
        summary.append(("Psychosis risk / Prodrome", "Elevated"))
    if cog_label != "No major cognitive concerns noted":
        summary.append(("Cognitive impairment", cog_label))
    if motor_label != "Normal":
        summary.append(("Motor/parkinsonism", motor_label))

    # Show user-friendly output
    st.markdown("---")
    st.header("📋 Screening Summary")
    if not summary:
        st.success("No strong signals for the screened conditions were detected. This does NOT rule out anything — if you are concerned, please consult a clinician.")
    else:
        st.warning("Potential concerns flagged (screening-level):")
        for cond, note in summary:
            st.write(f"- **{cond}** — {note}")

    st.markdown("### Detailed scores")
    st.write({
        "PHQ-9": phq_total,
        "GAD-7": gad_total,
        "MDQ_symptoms": mdq_total,
        "PQB_short": pqb_total,
        "Memory_recall": mem_score,
        "Verbal_fluency_count": vf_score,
        "Clock_score": clock_score,
        "Taps_10s": taps
    })

    # Save record option
    if st.button("Save my assessment (anonymous)"):
        rec = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "phq9": phq_total,
            "phq9_item9": phq9_item,
            "gad7": gad_total,
            "mdq_symptoms": mdq_total,
            "mdq_positive": mdq_positive,
            "pqb": pqb_total,
            "mem_score": mem_score,
            "vf_score": vf_score,
            "clock_score": clock_score,
            "taps": taps,
            "summary": ";".join([f"{c}:{n}" for c,n in summary]) or "No major signals"
        }
        save_result(rec)
        st.success("✅ Assessment saved.")

    # Final advice block (non-judgmental)
    st.markdown("---")
    st.header("Next steps (recommended)")
    st.write("""
    - These results are **screening-level only** and **not diagnostic**.
    - If any moderate/high scores or urgent flags occurred, make an appointment with a primary care doctor or mental health professional.
    - For cognitive concerns, ask your clinician about standardized cognitive testing (MoCA/MMSE), neuropsychology or neurology referral.
    - For motor concerns (tremor/bradykinesia), consider a neurology appointment.
    - If you have suicidal thoughts or plan, seek emergency services or crisis lines immediately.
    """)
