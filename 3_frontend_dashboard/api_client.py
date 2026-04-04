import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

MATH_NODE_URL = os.getenv("MATH_NODE_IP", "http://10.50.94.224:8000")
AI_NODE_URL = os.getenv("AI_NODE_IP", "http://10.50.94.31:8001")

# Standardize timeout for hackathon responsiveness
TIMEOUT = 15 

def fetch_math_score(gstin: str) -> dict:
    """Fetches the XGBoost score and SHAP values from the Ubuntu Math Engine."""
    try:
        response = requests.get(f"{MATH_NODE_URL}/score/{gstin}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"Math Engine Error: {str(e)}"}

def fetch_fraud_alerts(gstin: str) -> dict:
    """Checks the Ubuntu Math Engine for circular UPI network fraud."""
    try:
        response = requests.get(f"{MATH_NODE_URL}/check_fraud/{gstin}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"Fraud Detection Error: {str(e)}"}

def fetch_founder_risk(founder_data: dict) -> dict:
    """Sends founder profile to the Mac AI Hub for local LLM character analysis."""
    try:
        response = requests.post(f"{AI_NODE_URL}/founder_risk", json=founder_data, timeout=TIMEOUT * 2)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"AI Founder Analysis Error: {str(e)}"}

def fetch_shap_explanation(shap_data: dict) -> dict:
    """Sends raw SHAP math to the Mac AI Hub for plain-English translation."""
    try:
        response = requests.post(f"{AI_NODE_URL}/explain_shap", json=shap_data, timeout=TIMEOUT * 3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"AI SHAP Translation Error: {str(e)}"}

def fetch_market_swot(industry_context: dict) -> dict:
    """Sends sanitized sector data to the Mac AI Hub (which routes to Cloud)."""
    try:
        response = requests.post(f"{AI_NODE_URL}/market_swot", json=industry_context, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": f"Cloud SWOT Error: {str(e)}"}

if __name__ == "__main__":
    # Quick debug test block
    print(f"Loaded Math Node: {MATH_NODE_URL}")
    print(f"Loaded AI Node: {AI_NODE_URL}")
    print("\nTest Math Fetch for 'DUMMY_GSTIN':")
    print(fetch_math_score("DUMMY_GSTIN"))