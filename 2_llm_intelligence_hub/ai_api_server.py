from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# Import your custom logic from Files 5 & 6
from llama_cpp_client import LlamaIntelligenceHub
from cloud_router import CloudIntelligence

app = FastAPI(title="MSME Intelligence Router (Node 2)")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In hackathon, '*' is fine for Tailscale nodes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hubs
llm_hub = LlamaIntelligenceHub()
cloud_hub = CloudIntelligence()

# --- DATA MODELS ---

class FounderData(BaseModel):
    pan_status: str
    failed_businesses: int
    dependents: int

class ShapData(BaseModel):
    shap_dict: dict
    base_score: float

# --- ENDPOINTS ---

@app.post("/founder_risk")
async def get_founder_risk(data: FounderData):
    """Local LLM Analysis: Private founder history."""
    prompt = (
        f"Analyze this founder profile for a 24-hour credit decision: "
        f"PAN Status: {data.pan_status}, "
        f"Previous Business Failures: {data.failed_businesses}, "
        f"Number of Dependents: {data.dependents}. "
        "Briefly state the character risk level."
    )
    
    # We use the local hub to keep PII off the cloud
    result = await llm_hub.get_reasoning(prompt, system_prompt="You are a Credit Risk Officer. Be concise.")
    return result

@app.post("/explain_shap")
async def explain_shap_impact(data: ShapData):
    """Local LLM Analysis: Translating ML math to Banking English."""
    prompt = (
        f"The MSME has a credit score of {data.base_score}. "
        f"The machine learning model identifies these impacts (SHAP): {data.shap_dict}. "
        "Translate these technical values into 3 bullet points for a bank manager's report."
    )
    
    # Local reasoning is faster for small technical prompts
    result = await llm_hub.get_reasoning(prompt)
    return result

@app.post("/market_swot")
async def get_market_swot(industry_context: dict):
    """Cloud Analysis: Public sector trends (No PII)."""
    try:
        # Calls the Google/Anthropic Cloud Router
        swot_markdown = cloud_hub.generate_market_swot(industry_context)
        return {"swot": swot_markdown}
    except Exception as e:
        logger.error(f"Cloud SWOT failed: {e}")
        raise HTTPException(status_code=500, detail="Cloud API error")

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so Node 1 (Ubuntu) and Node 3 (Windows) can connect
    uvicorn.run(app, host="0.0.0.0", port=8001)