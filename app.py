import streamlit as st
from PIL import Image
import pandas as pd
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="ATHENANET:Neuropsy Early Screening App",
    page_icon="🧠",
    layout="wide",
)

# ---------------------- SIDEBAR DESIGN ----------------------
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=180)

st.sidebar.markdown("""
## 🧠 Neuropsy Early Screening
Your personal tool for early detection of  
**neuropsychiatric symptoms**,  
cognitive changes, and subtle early-stage  
mental-health patterns.

---

### 📂 Navigation
""")

# Auto-detect Streamlit page files:
if st.sidebar.button("🧠 Assessments"):
    st.switch_page("pages/1_🧩_Assessments.py")
if st.sidebar.button("📊 Dashboard"):
    st.switch_page("pages/2_📊_Dashboard.py")
if st.sidebar.button("🤖 Model Predictions"):
    st.switch_page("pages/3_🧠_Model_Predictions.py")
# st.sidebar.page_link("pages/4_⚙️_Admin_Controls.py", label="⚙️ Admin Controls")

st.sidebar.markdown("---")
st.sidebar.info("🔬 *Early screening only — not a diagnosis*")

# ---------------------- MAIN CONTENT ----------------------
st.title("🧠 Neuropsy Early Screening System")
st.subheader("Early Detection • Mental Health • Cognitive Patterns")

st.write("""
Welcome to the **Neuropsy Early Screening System** —  
an intelligent platform designed to detect **early signals**,  
patterns, and subtle markers of:

- Depression  
- Anxiety disorders  
- Bipolar spectrum patterns  
- Psychosis prodromal symptoms  
- Cognitive impairment (MCI / early dementia concerns)  
- Parkinsonian motor signs  

Your data is processed **locally**, stays **anonymous**,  
and provides you with **insights & next steps**.
""")

st.markdown("---")

# ---------------------- LAST ASSESSMENT SNAPSHOT ----------------------
st.subheader("📋 Your Latest Assessment Summary")

data_path = "data/user_assessments.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)

    if len(df) > 0:
        last = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🧠 PHQ-9 (Depression)", last["phq9"])
        col2.metric("😰 GAD-7 (Anxiety)", last["gad7"])
        col3.metric("🔥 MDQ Symptoms", last["mdq_symptoms"])
        col4.metric("👁 PQ-B Risk", last["pqb"])

        st.success("Your last assessment has been loaded successfully. View the Dashboard for details.")
    else:
        st.warning("No previous assessments found.")
else:
    st.warning("No assessment data found. Please complete your first assessment!")

st.markdown("---")

# ---------------------- QUICK START CARDS ----------------------
#'''
#st.subheader("🚀 Get Started Quickly")

#c1, c2, c3 = st.columns(3)

#with c1:
 #   st.markdown("### 🧠 Assess Yourself")
  #  st.write("Run the interactive **Neuropsy Assessment** to screen for early symptoms.")
   # st.page_link("pages/1_🧩_Assessments.py", 
    #label="Start Assessment")

#with c2:
#    st.markdown("### 📊 View Insights")
#    st.write("See your data visualized beautifully in the **Dashboard**.")
#    st.page_link("pages/2_📊_Dashboard.py", 
#    label="Open Dashboard")

#with c3:
    st.markdown("### 🤖 AI Predictions")
    st.write("Use AI models to analyze text and emotional patterns.")
    st.page_link("pages/3_🧠_Model_Predictions.py", 
    label="AI Predictions")

#st.markdown("---")
#'''
# ---------------------- FOOTER ----------------------
st.markdown("""
<div style='text-align:center; color:gray; margin-top:40px;'>
🧠 <b>Neuropsy Early Screening System</b> —  
Not a medical device. For awareness & early pattern detection only.  
Seek a licensed professional for diagnosis or treatment.
</div>
""", unsafe_allow_html=True)
