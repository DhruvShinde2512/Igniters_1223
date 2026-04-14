# MSME Credit Copilot

**A Privacy-First, Hybrid-AI Underwriting Engine for MSME Lending.**

## The Problem

Traditional banks underwrite Micro, Small, and Medium Enterprises (MSMEs) based on rigid historical data (GST filings, CIBIL scores). This ignores crucial "soft signals" like market reputation, local demand, and operational stability. While LLMs are perfect for analyzing these unstructured soft signals, banks cannot send highly sensitive PII and financial transaction logs to Cloud APIs (OpenAI/Claude) due to strict data compliance and privacy laws.

## Our Solution

The MSME Credit Copilot utilizes a Context-Aware LLM Router deployed across a localized mesh network.

We built a decentralized, multi-node architecture that intelligently routes analytical tasks based on data sensitivity:

1. The Math Engine: Traditional Machine Learning (XGBoost) processes raw UPI velocity and GST timeliness to generate a baseline credit score and SHAP (feature importance) values.
2. Local AI (Strict Privacy): Highly sensitive data (UPI transaction logs, PAN details, Founder liabilities) is analyzed 100% locally using dedicated Local AI Hardware running `llama.cpp`. It detects fraud, assesses founder risk, and translates complex SHAP math into plain-English banking advisory reports with zero data leakage.
3. Cloud AI (Macro-Intelligence): Sanitized, non-PII data (Industry sector, local vs. global buyer market) is routed to Cloud APIs to generate large-scale Market Scalability and SWOT analyses.

## Key Features

- Hybrid Scoring Engine: Fuses traditional XGBoost risk scoring with LLM-extracted sentiment and market analysis.
- Zero-Leakage AI: All Personally Identifiable Information (PII) stays on-premise.
- Real-Time Fraud Detection: Uses NetworkX graph traversal to instantly detect circular UPI transaction loops (e.g., A to B to C to A) indicating artificially inflated revenue.
- Plain-English Underwriting: AI translates complex mathematical weights (SHAP) into readable executive summaries for Loan Officers.
- Mesh Architecture: Computations are distributed across physical hardware nodes for maximum efficiency and parallel processing.

---

## Architecture & Tech Stack

Our system is deployed across a localized Mesh Network (via Tailscale) to separate concerns and maximize hardware utilization:

- Node 1 (Data & Math): Dedicated to synthetic data generation, XGBoost training, and API serving for mathematical scoring.
- Node 2 (Local AI Hardware): Dedicated to high-speed, local LLM inference and serving as the intelligent routing hub.
- Node 3 (UI Presentation): Dedicated to rendering the interactive Streamlit dashboard for the end-user.

Tech Stack:

- Frontend: Streamlit, Plotly
- Backend: FastAPI, Python, Uvicorn
- Data & ML: Pandas, XGBoost, SHAP, NetworkX
- AI & LLMs: `llama.cpp` (Local GGUF), GenAI SDKs

---

## Repository Structure

```text
msme-credit-copilot/
│
├── 1_data_and_math_engine/      # Node 1: Traditional ML & Math Server
│   ├── data_generator.py        # Generates synthetic GST, UPI, and JSON profiles
│   ├── xgboost_scorer.py        # Calculates 300-900 score and extracts SHAP values
│   └── math_api_server.py       # FastAPI server
│
├── 2_llm_intelligence_hub/      # Node 2: Local AI Hardware Hub
│   ├── local_model.gguf         # Local LLM weights
│   ├── ai_api_server.py         # FastAPI server routing Local vs Cloud
│   └── prompt_templates/        # Specialized prompts for financial extraction
│
└── 3_frontend_dashboard/        # Node 3: The Loan Officer UI
    ├── app.py                   # Streamlit dashboard
    ├── api_client.py            # Connects to Node 1 and Node 2 via Mesh
    └── .env                     # Contains Tailscale IPs for network routing
```
