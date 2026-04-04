import aiohttp
import json
import asyncio
from loguru import logger
from typing import Dict, Any, Optional

class LlamaIntelligenceHub:
    def __init__(self, host: str = "http://127.0.0.1", port: int = 8080):
        self.url = f"{host}:{port}/completion"
        logger.info(f"LLM Hub initialized for {self.url}")

    async def get_reasoning(
        self, 
        prompt: str, 
        system_prompt: str = "You are a senior credit risk analyst. Output ONLY valid JSON.",
        temperature: float = 0.2, 
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Sends a request to the local llama.cpp server.
        Uses a constrained grammar approach via the system prompt and JSON formatting.
        """
        
        # Construct the instruction following template (standard for Llama-3/Qwen)
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        payload = {
            "prompt": full_prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
            "stop": ["<|im_end|>", "</s>"],
            # llama.cpp specific: force JSON response if supported by your build
            "json_schema": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "risk_level": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                    "recommendation": {"type": "string"}
                },
                "required": ["analysis", "risk_level", "recommendation"]
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        raw_data = await response.json()
                        content = raw_data.get("content", "{}")
                        # Parse the string content into a dict
                        return json.loads(content)
                    else:
                        logger.error(f"LLM Server Error: {response.status}")
                        return {"error": "Server unreachable"}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}

async def test_run():
    hub = LlamaIntelligenceHub()
    # Mocking a SHAP-based prompt
    test_prompt = "Analyze MSME GSTIN-99. Score: 450. Top negative impact: GST_Timeliness (-80). Top positive: PAT_Margin (+20)."
    result = await hub.get_reasoning(test_prompt)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    asyncio.run(test_run())