import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# Point this to Tab 1 (your local llama.cpp server)
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

class LocalLLMHub:
    async def get_reasoning(self, prompt: str, system_prompt: str = "You are a Credit Risk Officer.", max_tokens: int = 512):
        """Routes requests directly to the local Llama model on your M4."""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2 # Low temperature for analytical consistency
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(LLAMA_SERVER_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"analysis": data["choices"][0]["message"]["content"]}
                    else:
                        error_text = await response.text()
                        return {"error": f"Local LLM Error: {error_text}"}
            except Exception as e:
                return {"error": f"Could not reach llama.cpp server: {str(e)}"}

app = FastAPI(title="MSME Intelligence Router (Node 2 - 100% Air-gapped)")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the 100% Local Hub
llm_hub = LocalLLMHub()

# --- DATA MODELS ---
class FounderData(BaseModel):
    pan_status: str
    failed_businesses: int
    dependents: int

class ShapData(BaseModel):
    shap_dict: dict
    base_score: float

class ChatData(BaseModel):
    prompt: str
    context: dict

# --- ENDPOINTS ---

@app.post("/founder_risk")
async def analyze_founder(data: dict):
    profile_text = str(data)
    prompt = f"Here is the founder's profile data: {profile_text}. Please analyze."
    
    system_rules = (
        "You are a meticulous Credit Risk Officer. Provide a highly detailed, "
        "comprehensive 3-paragraph psychological and financial analysis of this founder. "
        "Elaborate deeply on potential risks, operational strengths, and red flags. Do not be brief."
    )
    
    result = await llm_hub.get_reasoning(prompt, system_prompt=system_rules, max_tokens=800)
    return result

@app.post("/explain_shap")
async def explain_shap_impact(data: ShapData):
    prompt = (
        f"The MSME has a credit score of {data.base_score}. "
        f"The machine learning model identifies these impacts (SHAP): {data.shap_dict}. "
        "Translate these technical values into 3 bullet points for a bank manager's report."
    )
    
    result = await llm_hub.get_reasoning(prompt)
    return result

@app.post("/market_swot")
async def get_market_swot(industry_context: dict):
    """Now completely powered by your local M4 chip instead of the Cloud!"""
    prompt = f"Analyze this industry context and provide a brief SWOT analysis: {str(industry_context)}"
    system_rules = "You are a Market Intelligence Analyst. Provide a highly professional SWOT analysis in Markdown format."
    
    # We ask the local LLM for the SWOT
    result = await llm_hub.get_reasoning(prompt, system_prompt=system_rules, max_tokens=600)
    
    if "error" in result:
        logger.error(f"Local SWOT failed: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])
        
    # Format it so the Windows frontend receives exactly what it expects
    return {"swot": result.get("analysis", "No SWOT generated.")}

@app.post("/chat")
async def copilot_chat(data: ChatData):
    condensed_context = str(data.context)[:1000] 
    full_prompt = f"Context about the MSME: {condensed_context}\n\nThe Loan Officer asks: {data.prompt}\n\nProvide a professional, multi-sentence answer based on the context."
    
    result = await llm_hub.get_reasoning(full_prompt, max_tokens=512)
    return result


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so Node 1 (Ubuntu) and Node 3 (Windows) can connect
    uvicorn.run(app, host="0.0.0.0", port=8001)