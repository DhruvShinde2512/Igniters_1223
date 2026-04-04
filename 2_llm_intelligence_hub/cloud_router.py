import google.generativeai as genai
import os
import json
from loguru import logger

# --- CONFIGURATION ---
# Set your API Key in your terminal: export GOOGLE_API_KEY='your_key'
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

class CloudIntelligence:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Cloud Router active using {model_name}")

    def generate_market_swot(self, industry_context: dict) -> str:
        """
        Generates a sector-specific SWOT analysis.
        Input is sanitized (No PII, just sector and market types).
        """
        # Data Sanitization Check (Ensuring no IDs or specific names are leaked)
        sanitized_context = {
            "sector": industry_context.get("sector"),
            "market": industry_context.get("buyer_market_type"),
            "supply_chain": industry_context.get("supply_chain_dependency")
        }

        prompt = f"""
        Perform a professional SWOT analysis for an MSME in the following context:
        - Sector: {sanitized_context['sector']}
        - Market Type: {sanitized_context['market']}
        - Supply Chain Dependency: {sanitized_context['supply_chain']}

        Requirements:
        1. Focus on current 2026 economic trends in India.
        2. Format the output in clean Markdown.
        3. Use bullet points for each quadrant (Strengths, Weaknesses, Opportunities, Threats).
        4. Keep it concise for a bank manager's dashboard.
        """

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text
            return "SWOT Analysis unavailable at this time."
        except Exception as e:
            logger.error(f"Cloud API Error: {e}")
            return "Error connecting to Market Intelligence Cloud."

if __name__ == "__main__":
    # Mock sanitized data from your profiles
    mock_context = {
        "sector": "Textiles",
        "buyer_market_type": "Export",
        "supply_chain_dependency": "High"
    }
    
    router = CloudIntelligence()
    swot_md = router.generate_market_swot(mock_context)
    print("--- GENERATED MARKET SWOT ---")
    print(swot_md)