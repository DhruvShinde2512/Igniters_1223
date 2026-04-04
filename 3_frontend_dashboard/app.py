import streamlit as st
import plotly.graph_objects as go
import time

# Import the API client we just built
import api_client

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MSME Credit Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# --- UI HELPER FUNCTIONS ---
def create_gauge_chart(score):
    """Generates a professional Plotly gauge for the credit score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Credit Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E3A8A"}, # Deep banking blue
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 500], 'color': '#FCA5A5'}, # High Risk (Red)
                {'range': [500, 700], 'color': '#FDE047'}, # Review (Yellow)
                {'range': [700, 900], 'color': '#86EFAC'}  # Low Risk (Green)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- SIDEBAR: CONTROLS & INPUTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60) # Placeholder Bank Logo
    st.title("Copilot Controls")
    st.markdown("---")
    
    gstin_input = st.text_input("🔍 Search GSTIN", placeholder="Enter 15-digit GSTIN...")
    
    st.markdown("### 📎 Supplementary Documents")
    uploaded_file = st.file_uploader("Upload ITR/Bank Statement (PDF)", type="pdf")
    if uploaded_file:
        st.success(f"{uploaded_file.name} queued for OCR.")
        
    analyze_btn = st.button("Run Deep Analysis", type="primary", use_container_width=True)

# --- MAIN LOGIC EXECUTION ---
if analyze_btn and gstin_input:
    # Clear previous chat when searching a new GSTIN
    st.session_state.chat_history = [] 
    
    with st.status("Initializing MSME Intelligence...", expanded=True) as status:
        st.write("📡 Fetching baseline math & fraud data from Node 1 (Ubuntu)...")
        math_data = api_client.fetch_math_score(gstin_input)
        fraud_data = api_client.fetch_fraud_alerts(gstin_input)
        
        if math_data.get("error"):
            status.update(label="Failed to reach Math Node", state="error")
            st.error(math_data["message"])
            st.stop()

        st.write("🧠 Translating SHAP values on Node 2 (Mac M4)...")
        shap_payload = {"shap_dict": math_data["shap_explanations"], "base_score": math_data["base_score"]}
        shap_explanation = api_client.fetch_shap_explanation(shap_payload)
        
        # Note: In a full app, you'd fetch the actual owner profile from Node 1. 
        # Here we mock the payload for the AI Hub.
        st.write("👤 Analyzing founder profile & market trends...")
        mock_founder = {"pan_status": "Verified", "failed_businesses": 1, "dependents": 3}
        founder_risk = api_client.fetch_founder_risk(mock_founder)
        
        mock_industry = {"sector": "Textiles", "buyer_market_type": "Export", "supply_chain_dependency": "High"}
        market_swot = api_client.fetch_market_swot(mock_industry)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        # Save to session state so UI doesn't reset
        st.session_state.current_analysis = {
            "math": math_data,
            "fraud": fraud_data,
            "shap": shap_explanation,
            "founder": founder_risk,
            "swot": market_swot
        }

# --- MAIN DASHBOARD AREA ---
st.title("🏦 Executive Dashboard")

if st.session_state.current_analysis:
    data = st.session_state.current_analysis
    
    # FRAUD WARNING BANNER
    if data["fraud"].get("is_flagged") or data["math"].get("fraud_penalty", 0) > 0:
        st.error("🚨 **CRITICAL ALERT:** This GSTIN is involved in a circular transaction loop (Possible fake turnover). Proceed with extreme caution.")
    
    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "👤 Founder Risk", "🌍 Market Intelligence"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.plotly_chart(create_gauge_chart(data["math"].get("final_score", 0)), use_container_width=True)
            
        with col2:
            st.markdown("### 🤖 Copilot Rationale")
            # Safely extract text from the AI response dict
            shap_text = data["shap"].get("analysis", data["shap"].get("recommendation", str(data["shap"])))
            st.info(shap_text)
            
            st.markdown("### ⚙️ Raw ML Drivers")
            st.json(data["math"].get("shap_explanations", {}))
            
    with tab2:
        st.markdown("### Founder Character Analysis")
        st.caption("Locally generated on Mac M4 Pro to ensure PII privacy.")
        founder_text = data["founder"].get("analysis", data["founder"].get("recommendation", str(data["founder"])))
        st.write(founder_text)
        
    with tab3:
        st.markdown("### Q3 Sector Outlook")
        st.caption("Generated via Cloud Intelligence Engine.")
        st.markdown(data["swot"].get("swot", "No SWOT data available."))

    st.markdown("---")

else:
    st.info("👈 Enter a GSTIN in the sidebar to begin the analysis.")

# --- BOTTOM: COPILOT CHAT UI ---
st.markdown("### 💬 Chat with Credit Copilot")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about this application (e.g., 'What is the biggest risk here?'):"):
    # Append user prompt
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Mock AI response (In a real scenario, you'd send this to the Mac Node)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(1) # Simulating latency
            if st.session_state.current_analysis and "risk" in prompt.lower():
                response = "Based on the generated profile, the highest immediate risk is the GST filing delays detected by the XGBoost model."
            else:
                response = "I am tracking the MSME profile. Let me know if you need specific calculations or risk breakdowns."
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})