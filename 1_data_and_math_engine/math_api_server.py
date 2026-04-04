from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import json
import os

# Import your custom modules from the previous files
from xgboost_scorer import MSMEScorer
from fraud_detector import FraudDetector

app = FastAPI(title="MSME Real-Time Math Engine")

# --- CORS CONFIGURATION ---
# This allows Node 3 (Windows) to fetch data without browser blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with Windows Node Tailscale IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
scorer = MSMEScorer()
try:
    scorer.train_model()
    logger.success("XGBoost Scorer ready.")
except Exception as e:
    logger.error(f"Scorer initialization failed: {e}")

detector = FraudDetector()

# --- ENDPOINTS ---

@app.get("/get_data/{gstin}")
async def get_business_data(gstin: str):
    """Fetches raw profile data from the mock JSON."""
    try:
        with open("./mock_data/business_master_profiles.json", "r") as f:
            profiles = json.load(f)
        
        business = next((p for p in profiles if p["business_id"] == gstin), None)
        if not business:
            raise HTTPException(status_code=404, detail="GSTIN not found")
        return business
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/score/{gstin}")
async def get_credit_score(gstin: str, apply_amnesty: bool = False):
    """Calculates Credit Score and SHAP values via Node 1 logic."""
    
    # 1. We pass the amnesty flag down to where the data actually lives
    score, shaps = scorer.get_score_and_shap(gstin, apply_amnesty=apply_amnesty)
    
    if score is None:
        raise HTTPException(status_code=404, detail="Scoring failed for this GSTIN")
    
    # Check for fraud to apply a penalty (Hackathon logic)
    fraud_report = detector.get_fraud_report()
    is_fraudulent = any(gstin in alert["nodes_involved"] for alert in fraud_report["alerts"])
    
    adjusted_score = score - 200 if is_fraudulent else score
    
    return {
        "gstin": gstin,
        "base_score": score,
        "fraud_penalty": 200 if is_fraudulent else 0,
        "final_score": max(300, adjusted_score),
        "shap_explanations": shaps
    }

@app.get("/check_fraud/{gstin}")
async def check_fraud(gstin: str):
    # Call your existing fraud detector logic
    is_fraud = check_circular_fraud(gstin) # (Assuming this is your function)
    
    if is_fraud:
        return {
            "is_flagged": True,
            # We add a mock representation of the network graph here
            "fraud_loop_edges": [
                {"from": gstin, "to": "27AAPCU3456D1Z2", "amount": "₹5,00,000"},
                {"from": "27AAPCU3456D1Z2", "to": "27BBNDM1122E1Z5", "amount": "₹4,80,000"},
                {"from": "27BBNDM1122E1Z5", "to": gstin, "amount": "₹4,50,000"} # Loops back!
            ]
        }
    return {"is_flagged": False, "fraud_loop_edges": []}

if __name__ == "__main__":
    import uvicorn
    # Run on 0.0.0.0 to be accessible via Tailscale IP
    uvicorn.run(app, host="0.0.0.0", port=8000)