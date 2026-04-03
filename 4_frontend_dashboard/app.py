"""
System Context Prefix:
I am building a Real-Time MSME Credit Scoring system for a hackathon. The architecture is split across multiple machines connected via Tailscale. The math and data run on an Ubuntu node. The AI reasoning runs on a MacBook M4 Pro using llama.cpp (hosted locally). The UI is a Streamlit dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import json

# Import API wrapper functions mapped to Ubuntu/MacBook Tailscale nodes
from api_client import (
    get_business_data,
    get_ml_score,
    check_fraud,
    extract_text_features,
    get_shap_explanation,
    Messages
)

# -----------------------------------------------------------------------------
# Configuration & FinTech Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ignisia: Real-Time MSME Credit Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Styling to give a sleek 'Fintech' look */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #6c757d;
    }
    .app-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stExpander>div>div>p {
        font-size: 1.1rem;
        color: #333333;
    }
    .sidebar-section {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = {}
if "credit_score" not in st.session_state:
    st.session_state.credit_score = None
if "shap_text" not in st.session_state:
    st.session_state.shap_text = ""
if "fraud_alert" not in st.session_state:
    st.session_state.fraud_alert = False
if "business_data" not in st.session_state:
    st.session_state.business_data = {}


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def render_credit_gauge(score: int):
    """Renders a responsive Plotly gauge chart for the calculated credit score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Calculated Credit Score", 'font': {'size': 20, 'color': '#1E3A8A'}},
        number={'font': {'size': 50, 'color': '#1E3A8A'}},
        gauge={
            'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E3A8A"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e9ecef",
            'steps': [
                {'range': [300, 500], 'color': '#dc3545'}, # High Risk (Red)
                {'range': [500, 700], 'color': '#ffc107'}, # Moderate (Yellow)
                {'range': [700, 900], 'color': '#198754'}  # Prime (Green)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# -----------------------------------------------------------------------------
# App Header
# -----------------------------------------------------------------------------
st.markdown("<h1 class='app-title'>🏦 Ignisia Underwriting Dashboard</h1>", unsafe_allow_html=True)
st.caption("Distributed inference via Tailscale | Data: Ubuntu Node | Reasoning: MacBook M4 Node")
st.divider()


# -----------------------------------------------------------------------------
# Sidebar: Input Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Applicant Query")
    st.markdown("Enter standard business identification to fetch the corporate profile.")
    
    gstin_input = st.text_input("GSTIN Identifier", placeholder="e.g. 29GGGGG1314R9Z6")
    
    st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
    st.subheader("Supplementary Documents")
    uploaded_pdf = st.file_uploader("Upload Bank Statements/Financials (PDF)", type=["pdf"])
    
    st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("Generate Credit Score", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------
if analyze_btn:
    if not gstin_input:
        st.sidebar.error("GSTIN is required to begin analysis.")
    else:
        # Step 1: Fetch Business Data (Ubuntu)
        with st.spinner("Connecting to Ubuntu Node: Fetching business ledger..."):
            biz_data = get_business_data(gstin_input)
            st.session_state.business_data = biz_data or {"revenue": "Unknown", "status": "Not Found"}
            
        # Step 2: Fraud Check (Ubuntu)
        with st.spinner("Connecting to Ubuntu Node: Evaluating fraud vectors..."):
            fraud_res = check_fraud({"gstin": gstin_input, "data": st.session_state.business_data})
            st.session_state.fraud_alert = fraud_res.get("alert", False) if fraud_res else False
            
        # Step 3: Feature Extraction (MacBook)
        with st.spinner("Connecting to MacBook Node: Extracting PDF features..."):
            features = {}
            if uploaded_pdf:
                # Placeholder: In reality, extract byte array or text from the uploaded PDF
                features = extract_text_features(f"Extracted content from {uploaded_pdf.name}")

        # Step 4: ML Credit Scoring (Ubuntu)
        with st.spinner("Connecting to Ubuntu Node: Running scoring models..."):
            payload = {"business": st.session_state.business_data, "features": features}
            score_res = get_ml_score(payload)
            # Default to 650 for demonstration if API isn't live yet
            st.session_state.credit_score = score_res.get("score", 650) if score_res else 650
            
        # Step 5: SHAP Extraction (MacBook)
        with st.spinner("Connecting to MacBook Node: Formulating SHAP rationales..."):
            shap_payload = {"score": st.session_state.credit_score, "data": payload}
            shap_res = get_shap_explanation(shap_payload)
            st.session_state.shap_text = shap_res.get("explanation", "The AI identified no overwhelming negative attributes. Credit risk appears balanced according to historical MSME parameters.") if shap_res else "No reasoning available from node."
            
        # Save session context for the Copilot chat
        st.session_state.context = {
            "gstin": gstin_input,
            "credit_score": st.session_state.credit_score,
            "fraud_alert": st.session_state.fraud_alert,
            "business_data": st.session_state.business_data
        }


# -----------------------------------------------------------------------------
# Main Dashboard Panel
# -----------------------------------------------------------------------------
if st.session_state.credit_score is not None:
    
    # 1. Conditional Fraud Warning Box
    if st.session_state.fraud_alert:
        st.error("🚨 **FRAUD ALERT:** High-risk indicators detected in ledger or filings. Immediate manual review required.", icon="⚠️")
    else:
        st.success("✅ **Clear Status:** No prominent negative fraud patterns detected.")
        
    st.markdown("---")
        
    # 2. Main Metrics & Credit Score Gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Profile Summary")
        if st.session_state.business_data:
            st.metric("Annual Revenue", str(st.session_state.business_data.get("revenue", "$450,000")))
            st.metric("Time in Business", str(st.session_state.business_data.get("vintage", "4 Years")))
            st.metric("Compliance Tier", str(st.session_state.business_data.get("compliance", "Tier 1 - Standard")))
        else:
            st.info("No structured profile data loaded.")
            
    with col2:
        st.plotly_chart(render_credit_gauge(st.session_state.credit_score), use_container_width=True)

    # 3. LLM-Generated SHAP Plain-English Explanations
    st.markdown("### 🧠 AI Underwriting Rationale")
    with st.expander("View Interpretable Credit Factors", expanded=True):
        st.info(st.session_state.shap_text)

    st.divider()


# -----------------------------------------------------------------------------
# Copilot Chat Interface (MacBook Node)
# -----------------------------------------------------------------------------
st.markdown("### 💬 Loan Officer Copilot")
st.caption("Ask the local MacBook M4 LLM node questions regarding this specific applicant.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("E.g. What are the key risk factors lowering this applicant's score?"):
    
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context via local llama.cpp..."):
            response = Messages(msg=prompt, context=st.session_state.context)
            
            if response and "reply" in response:
                reply_text = response["reply"]
            else:
                reply_text = "Sorry, I am unable to connect to the MacBook AI Node at this time."
                
            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
