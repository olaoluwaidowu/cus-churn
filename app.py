"""
MIT807: Artificial Intelligence & Its Business Applications
Group 1 — Telecom Customer Churn Prediction
Streamlit Application
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | GROUP 1",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Force all Streamlit-rendered markdown text to be visible */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #f1f5f9 !important;
}
.stMarkdown p, .stMarkdown li {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {
    color: #e2e8f0 !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
    color: #e2e8f0;
}

/* ── Section headers ── */
.section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(167,139,250,0.3);
    padding-bottom: 6px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #6d28d9 0%, #2563eb 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(109,40,217,0.4);
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px 0;
}
.hero p {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
}

/* ── Result card ── */
.result-churn {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(239,68,68,0.35);
}
.result-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(16,185,129,0.35);
}
.result-icon {
    font-size: 3.5rem;
    margin-bottom: 8px;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
}
.result-conf {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin-top: 4px;
}
.confidence-bar-wrap {
    background: rgba(255,255,255,0.15);
    border-radius: 50px;
    height: 12px;
    margin: 16px 0 6px 0;
    overflow: hidden;
}
.confidence-bar-fill-red {
    height: 12px;
    border-radius: 50px;
    background: linear-gradient(90deg, #f87171, #ef4444);
    transition: width 0.4s ease;
}
.confidence-bar-fill-green {
    height: 12px;
    border-radius: 50px;
    background: linear-gradient(90deg, #34d399, #10b981);
    transition: width 0.4s ease;
}

/* ── Insight pills ── */
.insight-pill {
    display: inline-block;
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 12.5px;
    font-weight: 500;
    margin: 4px 4px 4px 0;
}
.pill-red   { background: rgba(239,68,68,0.2);   border:1px solid #ef4444; color:#fca5a5; }
.pill-green { background: rgba(16,185,129,0.2);  border:1px solid #10b981; color:#6ee7b7; }
.pill-amber { background: rgba(245,158,11,0.2);  border:1px solid #f59e0b; color:#fcd34d; }
.pill-blue  { background: rgba(59,130,246,0.2);  border:1px solid #3b82f6; color:#93c5fd; }

/* ══ WIDGET TEXT — force all labels & values to be clearly visible ══ */

/* All widget labels (selectbox, slider, number input, etc.) */
label,
.stSelectbox label,
.stSlider label,
.stNumberInput label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox > label > div,
div[class*="stSelectbox"] label,
div[class*="stSlider"] label,
div[class*="stNumberInput"] label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

/* Selectbox — the dropdown box itself */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.22) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
/* Selectbox selected value text — every possible node inside */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input,
div[data-baseweb="select"] p,
div[data-baseweb="select"] [class*="ValueContainer"],
div[data-baseweb="select"] [class*="ValueContainer"] *,
div[data-baseweb="select"] [class*="singleValue"],
div[data-baseweb="select"] [class*="placeholder"],
div[data-baseweb="select"] > div > div > div {
    color: #f1f5f9 !important;
}
/* Streamlit wraps selects in data-testid="stSelectbox" */
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div[class*="singleValue"] {
    color: #f1f5f9 !important;
}
/* Selectbox dropdown menu */
ul[data-baseweb="menu"],
div[data-baseweb="popover"] ul {
    background: #1e1b4b !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
div[data-baseweb="popover"] li,
ul[data-baseweb="menu"] li {
    color: #e2e8f0 !important;
    background: transparent !important;
}
div[data-baseweb="popover"] li:hover,
ul[data-baseweb="menu"] li:hover {
    background: rgba(167,139,250,0.2) !important;
}

/* Number input — dark background with light text (matches rest of form) */
div[data-testid="stNumberInput"] input,
.stNumberInput input {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.22) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}
/* Number input stepper buttons */
div[data-testid="stNumberInput"] button {
    color: #f1f5f9 !important;
    background: rgba(255,255,255,0.1) !important;
}

/* Slider — track label and value */
.stSlider > div { color: #e2e8f0 !important; }
[data-testid="stSlider"] div[class*="thumb"],
[data-testid="stSlider"] * { color: #e2e8f0 !important; }
/* Slider min/max tick labels */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #94a3b8 !important;
}

/* Help tooltip icon */
[data-testid="stWidgetLabel"] svg { color: #94a3b8 !important; }

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #e2e8f0 !important;
}

/* Form submit area & general p tags inside app */
.stApp p { color: #e2e8f0; }

/* Dataframe / table text */
[data-testid="stDataFrame"] * { color: #1e1b4b !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(90deg, #6d28d9, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(109,40,217,0.5);
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(109,40,217,0.65);
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1); }

/* ── Info boxes ── */
.stInfo, .stWarning, .stSuccess, .stError {
    border-radius: 10px;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_pipeline.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:3rem;'>📡</div>
        <div style='font-size:1.05rem; font-weight:700; color:#a78bfa;'>Churn Predictor</div>
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); margin-top:4px;'>MIT807 · GROUP 1</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1);'>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 About")
    st.markdown("""
    <div style='font-size:13.5px; color:rgba(255,255,255,0.7); line-height:1.7;'>
    This application uses a <b style='color:#a78bfa;'>Random Forest</b> machine learning model
    trained on the IBM Telco Customer Churn dataset to predict whether a customer is likely
    to churn (cancel their service).<br><br>
    Fill in the customer details and click <b style='color:#a78bfa;'>Predict Churn</b> to get
    an instant prediction with confidence level.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    <div style='font-size:13px; color:rgba(255,255,255,0.65); line-height:1.8;'>
    🌲 Algorithm: <b style='color:#c4b5fd;'>Random Forest</b><br>
    🌳 Estimators: <b style='color:#c4b5fd;'>500</b><br>
    📊 Test Accuracy: <b style='color:#c4b5fd;'>~81.4%</b><br>
    📁 Dataset: <b style='color:#c4b5fd;'>IBM Telco Churn</b><br>
    🔧 Pipeline: <b style='color:#c4b5fd;'>Scaler + Encoder + RF</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not found. Run `train_model.py` first.")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='badge'>📡 MIT807 · Artificial Intelligence & Its Business Applications</div>
    <h1>Telecom Customer Churn Predictor</h1>
    <p>Enter customer details below to predict the likelihood of churn using a trained Random Forest model · <b>GROUP 1</b></p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model pipeline not found. Please run `train_model.py` first to generate the model.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
with st.form("prediction_form"):

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT COLUMN ──────────────────────────────────────────────────────────
    with col_left:

        # Personal Info
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>👤 Personal Information</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c2:
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Account Info
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Account Details</div>", unsafe_allow_html=True)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12,
                                step=1, help="Number of months the customer has stayed")
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c2:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        st.markdown("</div>", unsafe_allow_html=True)

        # Charges
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💰 Charges</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0,
                                              value=65.0, step=0.5)
        with c2:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                            value=float(monthly_charges * tenure) if tenure > 0 else 65.0,
                                            step=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    with col_right:

        # Phone services
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📞 Phone Services</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        with c2:
            multiple_lines = st.selectbox("Multiple Lines",
                                          ["No", "Yes", "No phone service"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Internet services
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🌐 Internet Services</div>", unsafe_allow_html=True)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        c1, c2 = st.columns(2)
        with c1:
            online_security = st.selectbox("Online Security",
                                           ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup",
                                         ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection",
                                             ["No", "Yes", "No internet service"])
        with c2:
            tech_support = st.selectbox("Tech Support",
                                        ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV",
                                        ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies",
                                            ["No", "Yes", "No internet service"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit button ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Predict Churn")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    churn_prob = proba[1]
    safe_prob = proba[0]
    confidence = churn_prob if prediction == 1 else safe_prob

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Result")

    res_col, ins_col = st.columns([1, 1], gap="large")

    with res_col:
        if prediction == 1:
            bar_pct = int(churn_prob * 100)
            st.markdown(f"""
            <div class='result-churn'>
                <div class='result-icon'>⚠️</div>
                <div class='result-label'>HIGH CHURN RISK</div>
                <div class='result-conf'>This customer is likely to leave</div>
                <div class='confidence-bar-wrap'>
                    <div class='confidence-bar-fill-red' style='width:{bar_pct}%;'></div>
                </div>
                <div style='color:#fca5a5; font-size:2rem; font-weight:700;'>{bar_pct}%</div>
                <div style='color:rgba(255,255,255,0.6); font-size:13px;'>Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bar_pct = int(safe_prob * 100)
            st.markdown(f"""
            <div class='result-safe'>
                <div class='result-icon'>✅</div>
                <div class='result-label'>LOW CHURN RISK</div>
                <div class='result-conf'>This customer is likely to stay</div>
                <div class='confidence-bar-wrap'>
                    <div class='confidence-bar-fill-green' style='width:{bar_pct}%;'></div>
                </div>
                <div style='color:#6ee7b7; font-size:2rem; font-weight:700;'>{bar_pct}%</div>
                <div style='color:rgba(255,255,255,0.6); font-size:13px;'>Retention Probability</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Probability Breakdown</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; gap:12px; margin-top:8px;'>
            <div style='flex:1; background:rgba(16,185,129,0.15); border:1px solid #10b981;
                        border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:11px; letter-spacing:1px; text-transform:uppercase;
                            color:#6ee7b7; font-weight:600; margin-bottom:6px;'>Will NOT Churn</div>
                <div style='font-size:2rem; font-weight:700; color:#34d399;'>{round(safe_prob*100,1)}%</div>
            </div>
            <div style='flex:1; background:rgba(239,68,68,0.15); border:1px solid #ef4444;
                        border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:11px; letter-spacing:1px; text-transform:uppercase;
                            color:#fca5a5; font-weight:600; margin-bottom:6px;'>Will Churn</div>
                <div style='font-size:2rem; font-weight:700; color:#f87171;'>{round(churn_prob*100,1)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ins_col:
        st.markdown("<div class='section-card' style='height:100%;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔍 Risk Factor Insights</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13.5px; color:rgba(255,255,255,0.65); margin-bottom:12px;'>Key factors influencing this customer's churn risk:</div>", unsafe_allow_html=True)

        insights = []

        # Tenure
        if tenure <= 6:
            insights.append(("🔴", "New customer (≤6 months) — higher churn risk", "pill-red"))
        elif tenure >= 36:
            insights.append(("🟢", f"Loyal customer ({tenure} months tenure)", "pill-green"))
        else:
            insights.append(("🟡", f"Moderate tenure ({tenure} months)", "pill-amber"))

        # Contract
        if contract == "Month-to-month":
            insights.append(("🔴", "Month-to-month contract — most churn-prone", "pill-red"))
        elif contract == "One year":
            insights.append(("🟡", "One-year contract — moderate retention", "pill-amber"))
        else:
            insights.append(("🟢", "Two-year contract — high retention", "pill-green"))

        # Internet Service
        if internet_service == "Fiber optic":
            insights.append(("🔴", "Fiber optic service — associated with higher churn", "pill-red"))
        elif internet_service == "DSL":
            insights.append(("🟡", "DSL service — moderate churn tendency", "pill-amber"))
        else:
            insights.append(("🟢", "No internet service — lower churn risk", "pill-green"))

        # Payment method
        if payment_method == "Electronic check":
            insights.append(("🔴", "Electronic check — highest churn payment type", "pill-red"))
        else:
            insights.append(("🟢", f"Automatic/check payment — lower churn tendency", "pill-green"))

        # Senior citizen
        if senior_citizen == "Yes":
            insights.append(("🔴", "Senior citizen — tends to have higher churn", "pill-red"))

        # Partner & Dependents
        if partner == "No" and dependents == "No":
            insights.append(("🔴", "No partner or dependents — more likely to churn", "pill-red"))
        elif partner == "Yes" or dependents == "Yes":
            insights.append(("🟢", "Has partner/dependents — lower churn tendency", "pill-green"))

        # Tech support & security
        if online_security == "No" and tech_support == "No":
            insights.append(("🔴", "No online security or tech support", "pill-red"))
        elif online_security == "Yes" and tech_support == "Yes":
            insights.append(("🟢", "Has security & tech support services", "pill-green"))

        # Monthly charges
        if monthly_charges > 70:
            insights.append(("🔴", f"High monthly charges (${monthly_charges:.0f})", "pill-red"))
        elif monthly_charges < 30:
            insights.append(("🟢", f"Low monthly charges (${monthly_charges:.0f})", "pill-green"))
        else:
            insights.append(("🟡", f"Moderate monthly charges (${monthly_charges:.0f})", "pill-amber"))

        pills_html = ""
        for icon, text, css_class in insights:
            pills_html += f"<span class='insight-pill {css_class}'>{icon} {text}</span>"

        st.markdown(pills_html, unsafe_allow_html=True)

        # Recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💡 Recommended Action</div>", unsafe_allow_html=True)
        if prediction == 1:
            if churn_prob >= 0.75:
                rec = "🚨 <b>Urgent:</b> Customer is at very high risk. Offer immediate retention incentives — discount, contract upgrade, or dedicated support call."
                rec_color = "#ef4444"
            elif churn_prob >= 0.5:
                rec = "⚠️ <b>Proactive:</b> Schedule a customer satisfaction review. Consider offering a loyalty discount or upgrading service tier."
                rec_color = "#f59e0b"
            else:
                rec = "👀 <b>Monitor:</b> Customer shows mild churn signals. Follow up with a satisfaction survey and highlight service benefits."
                rec_color = "#f59e0b"
        else:
            if safe_prob >= 0.80:
                rec = "✅ <b>Retain & Upsell:</b> Highly loyal customer. Great candidate for premium service upgrades or referral programs."
                rec_color = "#10b981"
            else:
                rec = "✅ <b>Engage:</b> Customer is likely to stay. Maintain regular engagement and ensure service satisfaction."
                rec_color = "#10b981"

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.05); border-left:4px solid {rec_color};
                    border-radius:8px; padding:14px 16px; font-size:13.5px;
                    color:rgba(255,255,255,0.85); line-height:1.7;'>
            {rec}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:12px; padding:16px;
            border-top:1px solid rgba(255,255,255,0.08);'>
    MIT807: Artificial Intelligence & Its Business Applications &nbsp;·&nbsp;
    <b style='color:rgba(167,139,250,0.6);'>GROUP 1</b> &nbsp;·&nbsp;
    Customer Churn Prediction Project &nbsp;·&nbsp; Random Forest Model
</div>
""", unsafe_allow_html=True)
